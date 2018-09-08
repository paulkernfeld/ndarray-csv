# ndarray-csv

Easily read and write homogeneous CSV data to and from 2D ndarrays.

```rust
extern crate csv;
extern crate ndarray;
extern crate ndarray_csv;

use csv::{ReaderBuilder, WriterBuilder};
use ndarray::Array;
use ndarray_csv::{read, write};
use std::fs::File;
use std::io::{Read, Write};

fn main() {
    // Our 2x3 test array
    let array = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).into_shape((2, 3)).unwrap();

    // Write the array into the file.
    {
        let mut file = File::create("test.csv").expect("creating file failed");
        let mut writer = WriterBuilder::new().has_headers(false).from_writer(file);
        write(&array, &mut writer).expect("write failed");
    }

    // Read an array back from the file
    let mut file = File::open("test.csv").expect("opening file failed");
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
    let array_read = read((2, 3), &mut reader).expect("read failed");

    // Ensure that we got the original array back
    assert_eq!(array_read, array);
}
```

This project uses [cargo-make](https://sagiegurari.github.io/cargo-make/) for builds; to build,
run `cargo make all`.

To prevent denial-of-service attacks, do not read in untrusted CSV streams of unbounded length;
this can be implemented with `std::io::Read::take`.

License: MIT/Apache-2.0
