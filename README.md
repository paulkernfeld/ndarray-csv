# ndarray-csv

Easily read homogeneous CSV data into a 2D ndarray.

```rust
extern crate csv;
extern crate ndarray_csv;

use csv::ReaderBuilder;
use ndarray_csv::read;
use std::fs::File;

fn main() {
    let file = File::open("test.csv").expect("opening test.csv failed");
    let reader = ReaderBuilder::new().has_headers(false).from_reader(file);
    let array = read::<f64, _>((2, 3), reader).expect("read failed");
    assert_eq!(array.dim(), (2, 3));
}
```

This project uses [cargo-make](https://sagiegurari.github.io/cargo-make/) for builds; to build,
run `cargo make all`.

License: MIT/Apache-2.0
