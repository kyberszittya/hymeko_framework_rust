use std::error::Error;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, error, info, warn};
use arrow::array::{Float32Array, Int64Array};
use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::RecordBatch;
use hymeko::ir::hash::HashId;
use hymeko::ir::ir::Ir;
use hymeko::module_store::module_store::ModuleStore;
use hymeko::module_store::source_provider::MemProvider;
use hymeko::tensor::arrow_schema::schema_expansion_3d;
use hymeko::tensor::representations::tensor_coo::{CooEntry, TensorCoo};
use hymeko::traversal::hypergraphview::HyperGraphView;
use hymeko::tensor::tensor_val::{EdgeWScalar, RefValueExtractor, ScalarWeightExtractor};
use hymeko::util::real_parser::RealParser;
use hymeko::writers::cbor_writer::CborPayload;
use crate::service::{HymekoDaemon, PublishRequest}; // Import the struct from the service module

type ThreadSafeError = Box<dyn Error + Send + Sync>;
type ThreadSafeResult<T> = Result<T, ThreadSafeError>;

static REQUEST_SEQ: AtomicU64 = AtomicU64::new(1);

fn next_request_id() -> u64 {
    REQUEST_SEQ.fetch_add(1, Ordering::Relaxed)
}

fn etag_prefix_hex(etag: &[u8; 32]) -> String {
    let mut out = String::with_capacity(16);
    for b in &etag[..8] {
        let _ = write!(&mut out, "{b:02x}");
    }
    out
}

fn graph_name_from_ir(ir: &Ir) -> String {
    // Meta currently has no explicit name field, so derive a stable fallback label.
    ir.doc_hash
        .map(|hash| format!("graph-{}", etag_prefix_hex(&hash.0)))
        .unwrap_or_else(|| "AnonymousGraph".to_string())
}

fn tensor_to_arrow_bytes(
    tensor: TensorCoo<f32>,
    etag: &[u8; 32],
    graph_name: &str,
) -> Result<Vec<u8>, arrow::error::ArrowError> {
    // 1. Pivot the memory layout from AoS to SoA
    let soa = tensor.into_soa();
    // Metadata
    let mut metadata = std::collections::HashMap::new();
    metadata.insert("etag".to_string(), hex::encode(etag)); // Store the hash as hex
    metadata.insert("graph_name".to_string(), graph_name.to_string());

    // 2. Cast architecture-dependent usize to deterministic i64
    let k_array = Int64Array::from(soa.k.into_iter().map(|x| x as i64).collect::<Vec<_>>());
    let i_array = Int64Array::from(soa.i.into_iter().map(|x| x as i64).collect::<Vec<_>>());
    let j_array = Int64Array::from(soa.j.into_iter().map(|x| x as i64).collect::<Vec<_>>());
    let v_array = Float32Array::from(soa.v);

    // 3. Assemble the RecordBatch using your predefined schema
    let base_schema = schema_expansion_3d();
    let schema_owned = base_schema.as_ref().clone().with_metadata(metadata);
    let schema_arc = Arc::new(schema_owned);
    let batch = RecordBatch::try_new(
        schema_arc.clone(),
        vec![
            Arc::new(k_array),
            Arc::new(i_array),
            Arc::new(j_array),
            Arc::new(v_array),
        ],
    )?;

    // 4. Serialize into an IPC stream buffer
    let mut buffer = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buffer, &schema_arc)?;
        writer.write(&batch)?;
        writer.finish()?;
    }

    Ok(buffer)
}

impl HymekoDaemon {





    pub async fn handle_fast_path_ir(
        &self,
        payload: Vec<u8>,
        tx_pub: mpsc::Sender<PublishRequest>,
    ) -> Result<(), Box<dyn Error>> {
        let request_id = next_request_id();
        let service_name = self.config.service_name.clone();
        let payload_bytes = payload.len();
        info!(request_id, service = %service_name, source = "fast_path_ir", payload_bytes, "received fast-path IR payload");

        let (tx, rx) = oneshot::channel::<ThreadSafeResult<()>>();
        let graph_memory = Arc::clone(&self.graph_memory);
        let agg_cfg = Arc::clone(&self.agg_cfg);

        rayon::spawn(move || {
            match serde_cbor::from_slice::<CborPayload>(&payload) {
                Ok(cbor_payload) => {
                    let etag = cbor_payload.canon_hash.0;
                    let nnz = cbor_payload.ir.edges.len() as u64;
                    let ir_arc = Arc::new(cbor_payload.ir);
                    graph_memory.insert(etag, Arc::clone(&ir_arc));
                    info!(request_id, service = %service_name, source = "fast_path_ir", etag_prefix = %etag_prefix_hex(&etag), nnz, "IR stored in graph memory");
                    let extractor = ScalarWeightExtractor::default();
                    let hg_view = HyperGraphView::<f32, EdgeWScalar<f32>, f32>::from_ir(
                        &ir_arc, &agg_cfg, &extractor);

                    // Compute the sparse Star Expansion COO tensor (using f32 as the Real type)
                    let tensor = hymeko_hre::expansion::star_expansion_coo::<_, _, f32>(&hg_view);
                    let nnz = tensor.len() as u64;
                    let graph_name = graph_name_from_ir(&ir_arc);

                    // 3. ZERO-COPY BYTE CASTING
                    let tensor_data = tensor_to_arrow_bytes(tensor, &etag, graph_name.as_str())
                        .expect("Failed to encode Arrow IPC stream");

                    debug!(
                        request_id,
                        service = %service_name,
                        source = "fast_path_ir",
                        etag_prefix = %etag_prefix_hex(&etag),
                        nnz,
                        tensor_bytes = tensor_data.len(),
                        "decoded IR payload"
                    );

                    let req = PublishRequest { etag, nnz, tensor_data };
                    if let Err(e) = tx_pub.blocking_send(req) {
                        error!(request_id, service = %service_name, source = "fast_path_ir", error = %e, "failed to enqueue publish request");
                        let _ = tx.send(Err(format!("Failed to send to publisher actor: {}", e).into()));
                        return;
                    }
                    info!(request_id, service = %service_name, source = "fast_path_ir", "publish request enqueued");
                    let _ = tx.send(Ok(()));
                }
                Err(e) => {
                    error!(request_id, service = %service_name, source = "fast_path_ir", error = %e, "invalid IR CBOR payload");
                    let _ = tx.send(Err(format!("Invalid IR CBOR payload: {}", e).into()));
                }
            }
        });

        match rx.await {
            Ok(result) => result.map_err(|e| e as Box<dyn Error>),
            Err(_) => Err("Rayon worker hung up".into()),
        }
    }



    // Shared execution logic
    async fn execute_compilation(
        &self,
        dsl_source: String,
        tx_pub: mpsc::Sender<PublishRequest>
    ) -> Result<(), Box<dyn Error>> {
        let request_id = next_request_id();
        let service_name = self.config.service_name.clone();
        let source_len = dsl_source.len();
        let started = Instant::now();
        info!(request_id, service = %service_name, source = "execute_compilation", source_len, "starting compilation pipeline");

        let (tx, rx) = oneshot::channel::<ThreadSafeResult<()>>();
        let agg_cfg = Arc::clone(&self.agg_cfg);
        let graph_memory = Arc::clone(&self.graph_memory);
        rayon::spawn(move || {
            let compile_started = Instant::now();
            let mut fs = MemProvider::default();
            let query_path = PathBuf::from("incoming_query.hy");
            fs.insert_file(query_path.clone(), dsl_source);

            let parser = RealParser;
            let mut store = ModuleStore::new(fs, parser);

            let compiled = match store.compile(&query_path) {
                Ok(c) => c,
                Err(e) => {
                    error!(request_id, service = %service_name, source = "execute_compilation", error = ?e, "compilation failed");
                    let _ = tx.send(Err(format!("Compilation failed: {:?}", e).into()));
                    return;
                }
            };

            let compile_ms = compile_started.elapsed().as_secs_f64() * 1_000.0;
            let etag = compiled.canon_hash.0;
            drop(compiled);
            let owned_ir = store.take_last_ir().expect("Failed to extract IR from ephemeral compiler");
            let ir_arc = Arc::new(owned_ir);
            graph_memory.insert(etag, Arc::clone(&ir_arc));
            // 2. COMPUTE
            let extractor = ScalarWeightExtractor::default();
            let hg_view = HyperGraphView::<f32, EdgeWScalar<f32>, f32>::from_ir(
                &ir_arc, &agg_cfg, &extractor);
            let tensor = hymeko_hre::expansion::star_expansion_coo::<_, _, f32>(&hg_view);
            let nnz = tensor.len() as u64;
            let graph_name = graph_name_from_ir(&ir_arc);

            let tensor_data = tensor_to_arrow_bytes(tensor, &etag, graph_name.as_str())
                .expect("Failed to encode Arrow IPC stream");


            debug!(
                request_id,
                service = %service_name,
                source = "execute_compilation",
                etag_prefix = %etag_prefix_hex(&etag),
                nnz,
                tensor_bytes = tensor_data.len(),
                compile_ms,
                "compile output prepared"
            );

            let req = PublishRequest { etag, nnz, tensor_data };
            if let Err(e) = tx_pub.blocking_send(req) {
                error!(request_id, service = %service_name, source = "execute_compilation", error = %e, "failed to enqueue publish request");
                let _ = tx.send(Err(format!("Failed to send to publisher actor: {}", e).into()));
                return;
            }

            info!(request_id, service = %service_name, source = "execute_compilation", "publish request enqueued");
            let _ = tx.send(Ok(()));
        });

        match rx.await {
            Ok(result) => {
                let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
                match &result {
                    Ok(_) => info!(request_id, service = %self.config.service_name, source = "execute_compilation", elapsed_ms, "compilation pipeline completed"),
                    Err(e) => error!(request_id, service = %self.config.service_name, source = "execute_compilation", elapsed_ms, error = %e, "compilation pipeline failed"),
                }
                result.map_err(|e| e as Box<dyn Error>)
            }
            Err(_) => {
                error!(request_id, service = %self.config.service_name, source = "execute_compilation", "rayon worker hung up");
                Err("Rayon worker hung up".into())
            }
        }
    }

    pub async fn handle_utf8_query(&self, payload: Vec<u8>, tx: mpsc::Sender<PublishRequest>) -> Result<(), Box<dyn Error>> {
        let payload_bytes = payload.len();
        info!(service = %self.config.service_name, source = "handle_utf8_query", payload_bytes, "dispatching UTF-8 payload");
        let query_str = String::from_utf8(payload)?;
        self.execute_compilation(query_str, tx).await
    }

    pub async fn handle_cbor_query(&self, payload: Vec<u8>, tx: mpsc::Sender<PublishRequest>) -> Result<(), Box<dyn Error>> {
        let payload_bytes = payload.len();
        info!(service = %self.config.service_name, source = "handle_cbor_query", payload_bytes, "dispatching CBOR payload");
        let cbor_val: serde_cbor::Value = serde_cbor::from_slice(&payload)?;
        let query_str = match cbor_val {
            serde_cbor::Value::Text(s) => s,
            _ => {
                warn!(service = %self.config.service_name, source = "handle_cbor_query", "CBOR payload is not Text");
                return Err("Expected Text in CBOR".into());
            }
        };
        self.execute_compilation(query_str, tx).await
    }

    pub fn compile_to_ir_only(&self, dsl_source: String) -> Result<Arc<Ir>, Box<dyn Error + Send + Sync>> {
        // This is the first half of your existing execute_compilation logic
        let mut fs = MemProvider::default();
        let query_path = PathBuf::from("incoming_query.hy");
        fs.insert_file(query_path.clone(), dsl_source);

        let mut store = ModuleStore::new(fs, RealParser);
        store.compile(&query_path).map_err(|e| format!("Compile Error: {:?}", e))?;

        let ir = store.take_last_ir()?;
        Ok(Arc::new(ir))
    }

    pub fn deserialize_cbor_ir(&self, payload: &[u8]) -> Result<Arc<Ir>, Box<dyn Error + Send + Sync>> {
        let cbor_payload: CborPayload = serde_cbor::from_slice(payload)?;
        Ok(Arc::new(cbor_payload.ir))
    }

    pub fn expand_graph_clique(&self, ir: &Ir) -> Vec<u8> {
        let etag = ir.doc_hash.unwrap_or_else(|| {
            HashId([0; 32])
        }).0;

        let graph_name = ir.meta.as_ref()
            .map(|_m| "NamedGraph".to_string())
            .unwrap_or_else(|| "AnonymousGraph".to_string());

        let extractor = ScalarWeightExtractor::default();
        let hg_view = HyperGraphView::<f32, EdgeWScalar<f32>, f32>::from_ir(ir, &self.agg_cfg, &extractor);

        // Using your clique expansion algorithm here
        let tensor = hymeko_hre::expansion::clique_expansion_coo::<_, _, f32>(&hg_view);

        tensor_to_arrow_bytes(tensor, &etag, &graph_name).expect("Failed to encode Clique Arrow IPC stream")
    }

    pub fn serialize_ir_to_cbor(&self, ir: &Ir) -> Result<Vec<u8>, Box<dyn Error + Send + Sync>> {
        // Assuming you have #[derive(Serialize)] on your Ir struct
        let bytes = serde_cbor::to_vec(ir)?;
        Ok(bytes)
    }

    pub fn expand_graph(&self, ir: &Ir) -> Vec<u8> {
        let etag = ir.doc_hash.unwrap_or_else(|| {
            warn!(marker = "[-]", "IR reached execution without doc_hash. Using zeroed fallback.");
            HashId([0; 32]) // Fallback to zeroes
        }).0;
        let graph_name = graph_name_from_ir(ir);
        // 1. Setup the views using the extractor logic you already have
        let extractor = ScalarWeightExtractor::default();
        let hg_view = HyperGraphView::<f32, EdgeWScalar<f32>, f32>::from_ir(
            ir,
            &self.agg_cfg,
            &extractor
        );

        // 2. Perform the Star Expansion
        let tensor = hymeko_hre::expansion::star_expansion_coo::<_, _, f32>(&hg_view);

        // 3. Serialize to Arrow IPC bytes using your existing helper
        tensor_to_arrow_bytes(tensor, &etag, graph_name.as_str()).expect("Failed to encode Arrow IPC stream")
    }
}

