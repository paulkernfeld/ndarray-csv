# ndarray-csv

Easily read and write homogeneous CSV data to and from 2D ndarrays.

```rust
extern crate csv;
extern crate ndarray;
extern crate ndarray_csv;

use csv::{ReaderBuilder, WriterBuilder};
use ndarray::{array, Array2};
use ndarray_csv::{Array2Reader, Array2Writer};
use std::error::Error;
use std::fs::File;

fn main() -> Result<(), Box<dyn Error>> {
    // Our 2x3 test array
    let array = array![[1, 2, 3], [4, 5, 6]];

    // Write the array into the file.
    {
        let file = File::create("test.csv")?;
        let mut writer = WriterBuilder::new().has_headers(false).from_writer(file);
        writer.serialize_array2(&array)?;
    }

    // Read an array back from the file
    let file = File::open("test.csv")?;
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
    let array_read: Array2<u64> = reader.deserialize_array2((2, 3))?;

    // Ensure that we got the original array back
    assert_eq!(array_read, array);
    Ok(())
}
```

This project uses [cargo-make](https://sagiegurari.github.io/cargo-make/) for builds; to build,
run `cargo make all`.

To prevent denial-of-service attacks, do not read in untrusted CSV streams of unbounded length;
this can be implemented with `std::io::Read::take`.

License: MIT/Apache-2.0
