use ::std::fs::File;
use csv::Reader;
use serde::Deserialize;
use std::error::Error;
use std::path::Path;
use plotters::prelude::*;
use plotters::series::LineSeries;
use plotters::style::full_palette::{BLUE, RED};
use shared_lib::{predict, normalize, Model};
use owo_colors::OwoColorize;

#[derive(Debug, Deserialize)]
struct Field
{
    km: u64,
    price: f64,
}

fn parse<P: AsRef<Path>>(path: P) -> Result<Vec<Field>, Box<dyn Error>>
{
    let file = File::open(path)?;
    let mut reader = Reader::from_reader(file);
    let mut data = Vec::new();
    for result in reader.deserialize() {
        data.push(result?);
    }
    Ok(data)
}

fn mean_km(data: &[Field]) -> f64
{
    let sum: f64 = data.iter().map(|field| field.km as f64).sum();
    let count = data.len() as f64;
    sum / count
}

fn standard_deviation(mean: f64, data: &[Field]) -> f64
{
    let variance: f64 = data
        .iter()
        .map(|field| {
            let diff = field.km as f64 - mean;
            diff * diff
        })
        .sum::<f64>()
        / (data.len() as f64);
    variance.sqrt()
}

fn get_data_parameters(data: &[Field]) -> Model
{
    let mean = mean_km(&data);
    let standard_deviation = standard_deviation(mean, data);
    let (theta0, theta1) = (0.,0.);
    Model {theta0, theta1, mean, standard_deviation}
}

fn train(data: &[Field]) -> Model
{
    let mut model = get_data_parameters(data);
    let epochs = 1000;
    let learning_rate = 0.01;
    

    for _epoch in 0..epochs
    {
        let theta0_tmp = model.theta0 - learning_rate * data.iter().map(|f| {
            predict(f.km, &model) - f.price 
        }).sum::<f64>() / data.len() as f64;
        let theta1_tmp = model.theta1 - learning_rate * data.iter().map(|f| {
            (predict(f.km, &model) - f.price) * normalize(f.km, &model)
        }).sum::<f64>() / data.len() as f64;
        model.theta0 = theta0_tmp;
        model.theta1 = theta1_tmp;
    }
    model
}

fn draw(data: &[Field], model: &Model, path : &str) -> Result<(), Box<dyn Error>>
{
    let root = BitMapBackend::new(path, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let min_km = data.iter().map(|f| f.km as f64).fold(f64::INFINITY, |a, b| a.min(b));
    let max_km = data.iter().map(|f| f.km as f64).fold(f64::NEG_INFINITY, |a, b| a.max(b));
    let min_price = data.iter().map(|f| f.price).fold(f64::INFINITY, |a, b| a.min(b));
    let max_price = data.iter().map(|f| f.price).fold(f64::NEG_INFINITY, |a, b| a.max(b));
    
    let km_min = min_km * 0.9;
    let km_max = max_km * 1.1;
    let price_min = min_price * 0.9;
    let price_max = max_price * 1.1;
    
    let mut chart = ChartBuilder::on(&root)
        .caption("Linear Regression", ("sans-serif", 50).into_font())
        .margin(20)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(km_min..km_max, price_min..price_max)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(data.iter().map(|point| {
        let x = point.km as f64;
        let y = point.price;
        Circle::new((x, y), 5, BLUE.filled())
    }))?;

    let line_points: Vec<(f64, f64)> = (0..=100).map(|i| {
        let x = km_min + (km_max - km_min) * (i as f64 / 100.0);
        (x, predict(x as u64, model))
    }).collect();
    
    chart.draw_series(LineSeries::new(line_points, &RED))?
    .label("Regression Line")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}


fn main() -> Result<(), Box<dyn Error>>
{
    let path = "data/data.csv";
    let data = match parse(path){
        Ok(data) => {println!("Getting data from {}", path); data},
        Err(e) => {println!("Fail to get data from {}:{}", path, e); std::process::exit(1);}
    };
    
    let model = train(&data);
    let serialize = match serde_json::to_string(&model) {
        Ok(serialize) => serialize,
        Err(e) => {
            println!("Fail to serialize model data :{}", e);
            std::process::exit(1); 
        },
    };
    match std::fs::write("data/model.json", serialize){
        Ok(_a) => println!("Model successfully trained and saved in {}", path.bold().green()),
        Err(e) => println!("Fail to save the model in {}: {}", path.red(), e),
    };
    let graph_path = "data/model.png";
    match draw(&data, &model, graph_path) {
        Ok(_s) => println!("Graph of linear regression saved in {}", graph_path.bold().green()),
        Err(_e) => println!("Fail to save the graph in {}", graph_path.red()),
    };
    Ok(())
}
