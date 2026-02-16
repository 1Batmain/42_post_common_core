use shared_lib::{
    Field, 
    Model,
    parse,
    predict,
};


fn main() {

    let serialize: String = std::fs::read_to_string("data/model.json").unwrap();
    let model : Model = serde_json::from_str(&serialize).unwrap();
    let data : Vec<Field> = match parse("data/data.csv") {
        Ok(data)=> data,
        Err(e) => {
            eprintln!("Fail to parse data.csv: {}", e);
            std::process::exit(1);
        }
    };

    let mut predicted: f64;
    let mut expected: f64;
    let mut delta: f64 = 100.;

    for line in &data {
        predicted = predict(line.km, &model);
        expected = line.price;
        delta = (delta + (predicted * 100. / expected)) / 2. ;
    }

    println!("This model is accurate at {:.2}%", delta);

}