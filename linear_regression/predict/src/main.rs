use std::env;
use std::error::Error;
use serde::{Serialize, Deserialize};
use shared_lib::{predict, Model};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        println!("This program take the mileage of the car to estimate price");
        std::process::exit(1);
    }
    let mileage: f64 = args[1].parse()?;
    
    let serialized = std::fs::read_to_string("data/model.json")?;
    let model: Model = serde_json::from_str(&serialized)?;
    let price = predict(mileage, &model);
    println!("The estimated price for {mileage} is {price}");
    
    Ok(())
}
