use std::error::Error;
use std::sync::Arc;
use tokio::sync::oneshot;

use hymeko::tensor::shared_state::{calculate_required_bytes, ExpansionHeader};
use crate::service::{HymekoDaemon, PublishRequest}; // Import the struct from the service module

type ThreadSafeError = Box<dyn Error + Send + Sync>;
type ThreadSafeResult<T> = Result<T, ThreadSafeError>;

impl HymekoDaemon {
    pub async fn compute_expansion(
        &self,
        _etag: [u8; 32],
        publisher: Arc<iceoryx2::port::publisher::Publisher<iceoryx2::prelude::ipc::Service, [u8], ()>>,
    ) -> Result<(), Box<dyn Error>> {
        let (tx, rx) = oneshot::channel::<ThreadSafeResult<()>>();
        let interner = Arc::clone(&self.interner);

        // TODO tomorrow
        /*

        rayon::spawn(move || {
            let _lock = interner.read();

            // 1. Math stub: Assume we calculated an expansion of 1000 non-zero elements
            let nnz = 1000;

            // 2. Loan memory directly from iceoryx2 bypassing the heap
            let required_bytes = calculate_required_bytes(nnz);
            let mut sample = publisher.loan_slice_uninit(required_bytes)
                .expect("Failed to loan shared memory");

            // 3. Write the exact headers to the shared memory segment
            let payload = sample.payload_mut();
            let payload_ptr = payload.as_mut_ptr() as *mut u8;
            unsafe {
                let header_ptr = payload_ptr as *mut ExpansionHeader;
                (*header_ptr).nnz = nnz as u64;
                (*header_ptr).etag = _etag;
                // Tensor copying logic will go here
            }
            let initialized_sample = unsafe { sample.assume_init() };

            // 4. Instantly broadcast to PyTorch
            initialized_sample.send().expect("Failed to broadcast to PyTorch");

            let result: ThreadSafeResult<()> = Ok(());
            let _ = tx.send(result);
        });
        */

        match rx.await {
            Ok(result) => result.map_err(|e| e as Box<dyn Error>),
            Err(_) => Err("Rayon worker hung up".into()),
        }
    }

    pub async fn handle_query(
        &self,
        _payload: Vec<u8>,
        tx_pub: tokio::sync::mpsc::Sender<PublishRequest>,
    ) -> Result<(), Box<dyn Error>> {
        let (tx, rx) = oneshot::channel::<ThreadSafeResult<()>>();
        let interner = Arc::clone(&self.interner);

        rayon::spawn(move || {
            let _lock = interner.write().unwrap(); // Use write lock if parsing modifies the interner

            // 1. Math stub: Assume we calculated an expansion of 1000 non-zero elements
            let nnz = 1000;
            let etag = [0u8; 32];

            // 2. Build the exact memory layout of the tensor in a local Vec
            // For now, an empty Vec is fine to prove the pipeline connects.
            let tensor_data = Vec::new();

            // 3. Send the instruction back to the Tokio Main Thread Actor
            let req = PublishRequest { etag, nnz, tensor_data };
            if let Err(e) = tx_pub.blocking_send(req) {
                let _ = tx.send(Err(format!("Failed to send to publisher actor: {}", e).into()));
                return;
            }

            let _ = tx.send(Ok(()));
        });

        match rx.await {
            Ok(result) => result.map_err(|e| e as Box<dyn Error>),
            Err(_) => Err("Rayon worker hung up".into()),
        }
    }
}