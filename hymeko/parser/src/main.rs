mod ast; // Import your AST module

use parser::hymeko;
use std::env;
use std::fs;
use std::process;

// Args: path
fn main() {
    // 1. Get command line arguments
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <file_path>", args[0]);
        process::exit(1);
    }

    let file_path = &args[1];

    // 2. Read the file content
    let input = match fs::read_to_string(file_path) {
        Ok(content) => content,
        Err(e) => {
            eprintln!("❌ Failed to read file '{}': {}", file_path, e);
            process::exit(1);
        }
    };

    // 3. Initialize the parser for the "Start" rule
    let parser = hymeko::DescriptionParser::new();

    // 4. Parse and handle result
    match parser.parse(&input) {
        Ok(ast) => {
            println!("✅ Parse Successful!");
            println!("{:#?}", ast); // Pretty-print the AST
        },
        Err(e) => {
            println!("❌ Parse Failed:");
            println!("{:?}", e);
            process::exit(1);
        }
    }
}
