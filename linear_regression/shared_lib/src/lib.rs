use ::std::fs::File;
use csv::Reader;
use serde::{Serialize, Deserialize};
use std::path::Path;
use std::error::Error;

#[derive(Default, Debug, Deserialize, Serialize)]
pub struct Model
{
    pub theta0: f64,
    pub theta1: f64,
    pub mean: f64,
    pub standard_deviation: f64,
}

#[derive(Debug, Deserialize)]
pub struct Field
{
    pub km: u64,
    pub price: f64,
}

pub fn normalize(km: u64, model: &Model) -> f64
{
    (km as f64 - model.mean) / model.standard_deviation
}

pub fn predict(km: u64, model: &Model) -> f64
{
    let km = normalize(km, model);
    model.theta0 + km * model.theta1
}

pub fn parse<P: AsRef<Path>>(path: P) -> Result<Vec<Field>, Box<dyn Error>>
{
    let file = File::open(path)?;
    let mut reader = Reader::from_reader(file);
    let mut data = Vec::new();
    for result in reader.deserialize() {
        data.push(result?);
    }
    Ok(data)
}