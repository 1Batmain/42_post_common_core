use shared_lib::{
    Field,
    Model,
    parse,
    predict,
};


fn main() {

    let serialize: String = match std::fs::read_to_string("data/model.json"){
        Ok(ser) => ser,
        Err(_) => {
            println!("You must train model to test it !");
            std::process::exit(1);
        }
    };
    let model : Model = match serde_json::from_str(&serialize){
        Ok(m) => m,
        Err(e) => {
            println!("Error in the model.json spec file: {}", e);
            std::process::exit(1);
        }
    };
    let data : Vec<Field> = match parse("data/data.csv") {
        Ok(data)=> data,
        Err(e) => {
            eprintln!("Fail to parse data.csv: {}", e);
            std::process::exit(1);
        }
    };

    let mut predicted: f32;
    let mut expected: f32;
    let mut delta: f32 = 100.;

    for line in &data {
        predicted = predict(line.km, &model);
        expected = line.price;
        delta = (delta + (predicted * 100. / expected)) / 2. ;
    }

    println!("This model is accurate at {:.2}%", delta);

}
