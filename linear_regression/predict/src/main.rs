use std::env;
use std::error::Error;
use shared_lib::{predict, Model};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        println!("This program take the mileage of the car to estimate price");
        std::process::exit(1);
    }

    let path = "data/model.json";
    
    let serialized = match std::fs::read_to_string(path) {
        Ok(content) => { println!("Getting weights from {path}"); content },
        Err(e) => {eprintln!("Failed to read the file : {}", e); std::process::exit(1)},
    };
    let model: Model = match serde_json::from_str(&serialized) {
        Ok(model) => {println!("Model weights sets !"); model },
        Err(e) => {eprintln!("Failed to parse model weights, try to run the train program again : {}", e); std::process::exit(1) },
    };


    let mileage: u64 = match args[1].parse() {
        Ok(mileage) => mileage,
        Err(e) => { eprintln!("cant estimate price for \"{}\" because itsm", args[1]); std::process::exit(1) },
    };

    let price = predict(mileage, &model) as u64;
    println!("The estimated price for {mileage} miles is {price} $");

    Ok(())
}
