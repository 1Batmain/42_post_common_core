use colored::Colorize;
use shared_lib::{Model, predict};

use std::env;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();

    let path = match args.get(1) {
        Some(path) => path,
        None => "data/model.json",
    };

    let model: Model =
        match std::fs::read_to_string(path) {
            Ok(serialized) => {
                match serde_json::from_str(&serialized) {
                    Ok(model) => {
                        println!("Model weights sets !");
                        model
                    }
                    Err(e) => {
                        println!("Invalids model weights in file {}, run train program again :{}", path.bold(), e);
                        Model::default()
                    }
                }
            }
            Err(_) => {
                println!("No model found at {} you may run the train program for accurate results (result will always be 0)", path.bold());
                Model::default()
            }
        };

    println!("Enter a mileage to predict :");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap_or(0);
    let input = input.trim();

    let mileage: f32 =  match input.parse(){
        Ok(mileage) => mileage,
        Err(_e) => {
            eprintln!("cant estimate price for \"{}\" because its not a valid mileage number", input);
            std::process::exit(1)
        }
    };

    let price = predict(mileage, &model) as f32;
    println!(
        "The estimated price for {} miles is {} $",
        mileage.to_string().bold(),
        price.to_string().bold()
    );

    Ok(())
}
