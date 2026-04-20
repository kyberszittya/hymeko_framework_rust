//! stdio driver for the MCP server — reads newline-delimited JSON-RPC
//! requests from stdin, writes responses to stdout. This is the shape
//! Claude Code expects for a local MCP server configured in
//! `.claude/mcp.json`.

use hymeko_mcp::McpServer;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

#[tokio::main]
async fn main() -> std::io::Result<()> {
    let server = McpServer::new();

    let stdin = tokio::io::stdin();
    let mut reader = BufReader::new(stdin).lines();
    let mut stdout = tokio::io::stdout();

    while let Some(line) = reader.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }
        let response = server.handle_request(&line);
        stdout.write_all(response.as_bytes()).await?;
        stdout.write_all(b"\n").await?;
        stdout.flush().await?;
    }
    Ok(())
}
