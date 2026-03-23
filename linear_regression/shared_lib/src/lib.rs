use ::std::fs::File;
use csv::Reader;
use serde::{Serialize, Deserialize};
use std::path::Path;
use std::error::Error;

#[derive(Debug, Deserialize, Serialize)]
pub struct Model
{
    pub theta0: f32,
    pub theta1: f32,
    pub mean: f32,
    pub standard_deviation: f32,
}

impl Default for Model {
    fn default() -> Self {
        Self {
            theta0: 0.,
            theta1: 0.,
            mean: 0.,
            standard_deviation: 1.,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct Field
{
    pub km: f32,
    pub price: f32,
}

pub fn normalize(km: f32, model: &Model) -> f32
{
    (km as f32 - model.mean) / model.standard_deviation
}

pub fn predict(km: f32, model: &Model) -> f32
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
