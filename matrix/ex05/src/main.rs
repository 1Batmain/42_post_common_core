use colored::Colorize;
use matrix::{Tensor, Vector, angle_cos};

fn main() {
    println!("{}", "ex05 - angle_cos".bold().underline().green());

    let vnull = Vector::from([0.; 3]);
    let vx = Vector::from([3., 0., 0.]);
    let vy = Vector::from([0., 3., 0.]);
    let mut vy_neg = vy.clone();
    vy_neg.scl(-1.).expect("Scaling error");

    println!(
        "{}",
        "\n[Test 1] angle_cos(vnull, vx) should be Err"
            .bold()
            .yellow()
    );
    match angle_cos(&vnull, &vx) {
        Ok(v) => println!("unexpected Ok({v})"),
        Err(e) => println!("{}", format!("Ok: got Err(\"{e}\")").green()),
    }

    println!(
        "{}",
        "\n[Test 2] angle_cos(vx, vx) should be Ok(1.)"
            .bold()
            .yellow()
    );
    match angle_cos(&vx, &vx) {
        Ok(v) => println!(
            "{}",
            format!("result: {v} {}", if v == 1. { "✅" } else { "❌" }).green()
        ),
        Err(e) => println!("unexpected Err({e})"),
    }

    println!(
        "{}",
        "\n[Test 3] angle_cos(vy, -vy) should be Ok(-1.)"
            .bold()
            .yellow()
    );
    match angle_cos(&vy, &vy_neg) {
        Ok(v) => println!(
            "{}",
            format!("result: {v} {}", if v == -1. { "✅" } else { "❌" }).green()
        ),
        Err(e) => println!("unexpected Err({e})"),
    }

    println!("\nvy      = {}", vy);
    println!("vy_neg  = {}", vy_neg);
}
