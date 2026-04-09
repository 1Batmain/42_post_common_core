use colored::Colorize;
use matrix::{Tensor, Vector};

fn main() {
    println!("{}", "ex04 - Norms".bold().underline().green());

    let vx = Vector::from([3., 0., 0.]);
    let vy = Vector::from([0., 3., 0.]);
    let vz = Vector::from([0., 0., 3.]);
    let empty = Vector::<f32>::from([]);
    let zero = Vector::<f32>::from([0.; 3]);

    println!("\n{}", "Empty / zero vectors".bold().yellow());
    println!("empty.norm_1() -> {:?}", empty.norm_1());
    println!("empty.norm_2() -> {:?}", empty.norm_2());
    println!("empty.norm_3() -> {:?}", empty.norm_3());
    println!("zero.norm_1()  -> {:?}", zero.norm_1());
    println!("zero.norm_2()  -> {:?}", zero.norm_2());
    println!("zero.norm_3()  -> {:?}", zero.norm_3());

    assert!(empty.norm_1().is_none());
    assert!(empty.norm_2().is_none());
    assert!(empty.norm_3().is_none());
    assert!(zero.norm_1().is_none());
    assert!(zero.norm_2().is_none());
    assert!(zero.norm_3().is_none());

    println!("\n{}", "Axis vectors".bold().yellow());
    println!("vx -> {}", vx);
    println!("vy -> {}", vy);
    println!("vz -> {}", vz);

    assert_eq!(vx.norm_1(), Some(3.0));
    assert_eq!(vy.norm_1(), Some(3.0));
    assert_eq!(vz.norm_1(), Some(3.0));

    assert_eq!(vx.norm_2(), Some(3.0));
    assert_eq!(vy.norm_2(), Some(3.0));
    assert_eq!(vz.norm_2(), Some(3.0));

    assert_eq!(vx.norm_3(), Some(3.0));
    assert_eq!(vy.norm_3(), Some(3.0));
    assert_eq!(vz.norm_3(), Some(3.0));

    let vecteureu = Vector::from([-20., 10., -3.]);
    println!("\n{}", "Infinity norm case".bold().yellow());
    println!("vecteureu -> {}", vecteureu);
    println!("vecteureu.norm_3() -> {:?}", vecteureu.norm_3());
    assert_eq!(vecteureu.norm_3(), Some(20.));

    let mut v = vx.clone();
    v.add(&vy).unwrap();
    v.add(&vz).unwrap();
    println!("\n{}", "After vector addition".bold().yellow());
    println!("v = vx + vy + vz -> {}", v);
    println!("v.norm_1() -> {:?}", v.norm_1());
    assert_eq!(v.norm_1(), Some(9.0));

    let fix_norm = Vector::from([80198051.0; 3]);
    println!("\n{}", "Large values norm_2 case".bold().yellow());
    println!("fix_norm.norm_2() -> {:?}", fix_norm.norm_2());
    assert_eq!(fix_norm.norm_2(), Some(138907099.0));

    println!(
        "\n{}",
        "ex04 finished successfully ✅".bold().underline().green()
    );
}
