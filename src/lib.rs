//! Easily read and write homogeneous CSV data to and from 2D ndarrays.
//!
//! ```rust
//! extern crate csv;
//! extern crate ndarray;
//! extern crate ndarray_csv;
//!
//! use csv::{ReaderBuilder, WriterBuilder};
//! use ndarray::Array;
//! use ndarray_csv::{read, write};
//! use std::fs::File;
//! use std::io::{Read, Write};
//!
//! fn main() {
//!     // Our 2x3 test array
//!     let array = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).into_shape((2, 3)).unwrap();
//!
//!     // Write the array into the file.
//!     {
//!         let mut file = File::create("test.csv").expect("creating file failed");
//!         let mut writer = WriterBuilder::new().has_headers(false).from_writer(file);
//!         write(&array, &mut writer).expect("write failed");
//!     }
//!
//!     // Read an array back from the file
//!     let mut file = File::open("test.csv").expect("opening file failed");
//!     let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
//!     let array_read = read((2, 3), &mut reader).expect("read failed");
//!
//!     // Ensure that we got the original array back
//!     assert_eq!(array_read, array);
//! }
//! ```
//!
//! This project uses [cargo-make](https://sagiegurari.github.io/cargo-make/) for builds; to build,
//! run `cargo make all`.
extern crate csv;
#[cfg(test)]
#[macro_use]
extern crate matches;
#[cfg_attr(test, macro_use(array))]
extern crate ndarray;
extern crate serde;

use csv::{Reader, Writer};
use ndarray::Array2;
use std::io::{Read, Write};

#[derive(Debug)]
pub enum Error {
    Csv(csv::Error),
    TooFewRows {
        expected: usize,
        actual: usize,
    },
    // We don't want to read the whole file, so this only reports that there were too many rows
    TooManyRows {
        expected: usize,
    },
    TooFewColumns {
        at_row_index: usize,
        expected: usize,
        actual: usize,
    },
    // We don't want to read the whole row, so this only reports that there were too many columns
    TooManyColumns {
        at_row_index: usize,
        expected: usize,
    },
}

impl From<csv::Error> for Error {
    fn from(inner: csv::Error) -> Self {
        Error::Csv(inner)
    }
}

/// Read CSV data into a new ndarray with the given shape
pub fn read<A, R>(shape: (usize, usize), reader: &mut Reader<R>) -> Result<Array2<A>, Error>
where
    R: Read,
    A: Copy,
    for<'de> A: serde::Deserialize<'de>,
{
    // This is okay because this fn will return an Err when it is unable to fill the entire array.
    // Since this array is only returned when this fn returns an Ok, the end user will never be able
    // to read unitialized memory.
    let mut array = unsafe { Array2::uninitialized(shape) };
    let (n_rows, n_columns) = shape;
    let mut max_row_read: usize = 0;
    for (r, row) in reader.deserialize().enumerate() {
        if r >= n_rows {
            return Err(Error::TooManyRows { expected: n_rows });
        }

        let row_vec: Vec<A> = row?;
        let mut max_column_read: usize = 0;
        for (c, value) in row_vec.into_iter().enumerate() {
            if c >= n_columns {
                return Err(Error::TooManyColumns {
                    at_row_index: r,
                    expected: n_columns,
                });
            }

            array[(r, c)] = value;
            max_column_read = c
        }

        let n_columns_read = max_column_read + 1;
        if n_columns_read < n_columns {
            return Err(Error::TooFewColumns {
                at_row_index: r,
                expected: n_columns,
                actual: n_columns_read,
            });
        }

        max_row_read = r;
    }
    let n_rows_read = max_row_read + 1;
    if n_rows_read < n_rows {
        Err(Error::TooFewRows {
            expected: n_rows,
            actual: n_rows_read,
        })
    } else {
        Ok(array)
    }
}

/// Write this ndarray into CSV format
pub fn write<A, W>(array: &Array2<A>, writer: &mut Writer<W>) -> Result<(), csv::Error>
where
    A: serde::Serialize,
    W: Write,
{
    for row in array.outer_iter() {
        writer.serialize(row.as_slice())?;
    }
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::Error::*;
    use super::*;
    use csv::ReaderBuilder;
    use std::io::Cursor;
    use csv::WriterBuilder;

    fn in_memory_reader(content: &'static str) -> Reader<impl Read> {
        ReaderBuilder::new()
            .has_headers(false)
            .from_reader(Cursor::new(content))
    }

    fn test_reader() -> Reader<impl Read> {
        in_memory_reader("1,2,3\n4,5,6\n")
    }

    #[test]
    fn test_read_float() {
        let actual = read((2, 3), &mut test_reader()).unwrap();
        let expected = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_read_integer() {
        let actual = read((2, 3), &mut test_reader()).unwrap();
        let expected = array![[1, 2, 3], [4, 5, 6]];
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_read_csv_error() {
        assert_matches! {
            read::<i8, _>((2, 3), &mut in_memory_reader("1,2,3\n4,x,6\n")),
            Err(Csv(_))
        }
    }

    #[test]
    fn test_read_too_few_rows() {
        assert_matches! {
            read::<i8, _>((3, 3), &mut test_reader()),
            Err(TooFewRows { expected: 3, actual: 2})
        }
    }

    #[test]
    fn test_read_too_many_rows() {
        assert_matches! {
            read::<i8, _>((1, 3), &mut test_reader()),
            Err(TooManyRows { expected: 1 })
        }
    }

    #[test]
    fn test_read_too_few_columns() {
        assert_matches! {
            read::<i8, _>((2, 4), &mut test_reader()),
            Err(TooFewColumns { at_row_index: 0, expected: 4, actual: 3 })
        }
    }

    #[test]
    fn test_read_too_many_columns() {
        assert_matches! {
            read::<i8, _>((2, 2), &mut test_reader()),
            Err(TooManyColumns { at_row_index: 0, expected: 2 })
        }
    }

    #[test]
    fn test_write_ok() {
        let mut writer = WriterBuilder::new()
            .has_headers(false)
            .from_writer(vec![]);

        assert_matches! {
            write(&array![[1, 2, 3], [4, 5, 6]], &mut writer),
            Ok(())
        }
        assert_eq!(writer.into_inner().expect("flush failed"), b"1,2,3\n4,5,6\n");
    }

    #[test]
    fn test_write_err() {
        let destination: &mut [u8] = &mut [0; 8];
        let mut writer = WriterBuilder::new()
            .has_headers(false)
            .from_writer(Cursor::new(destination));

        // The destination is too short
        assert_matches! {
            write(&array![[1, 2, 3], [4, 5, 6]], &mut writer),
            Err(_)
        }
    }
}
