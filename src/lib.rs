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
//!
//! fn main() {
//!     // Our 2x3 test array
//!     let array = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).into_shape((2, 3)).unwrap();
//!
//!     // Write the array into the file.
//!     {
//!         let file = File::create("test.csv").expect("creating file failed");
//!         let mut writer = WriterBuilder::new().has_headers(false).from_writer(file);
//!         write(&array, &mut writer).expect("write failed");
//!     }
//!
//!     // Read an array back from the file
//!     let file = File::open("test.csv").expect("opening file failed");
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
//!
//! To prevent denial-of-service attacks, do not read in untrusted CSV streams of unbounded length;
//! this can be implemented with `std::io::Read::take`.
extern crate csv;
extern crate either;
#[macro_use]
extern crate failure;
#[cfg(test)]
#[macro_use]
extern crate matches;
#[cfg_attr(test, macro_use(array))]
extern crate ndarray;
extern crate serde;

use csv::{Reader, Writer};
use either::Either;
use failure::Error;
use ndarray::{Array1, Array2};
use std::io::{Read, Write};
use std::iter::once;

#[derive(Debug, Fail)]
pub enum ReadError {
    #[fail(
        display = "wrong number of rows: expected {}, actual {}",
        expected,
        actual
    )]
    NRows { expected: usize, actual: usize },

    #[fail(
        display = "wrong number of columns on row {}: expected {}, actual {}",
        at_row_index,
        expected,
        actual
    )]
    NColumns {
        at_row_index: usize,
        expected: usize,
        actual: usize,
    },
}

/// Read CSV data into a new ndarray with the given shape
pub fn read<A>(shape: (usize, usize), reader: &mut Reader<impl Read>) -> Result<Array2<A>, Error>
where
    A: Copy,
    for<'de> A: serde::Deserialize<'de>,
{
    let (n_rows, n_columns) = shape;

    let rows = reader.deserialize::<Vec<A>>();
    let values = rows.enumerate().flat_map(|(row_index, row)| match row {
        Err(e) => Either::Left(once(Err(Error::from(e)))),
        Ok(row_vec) => Either::Right(if row_vec.len() == n_columns {
            Either::Right(row_vec.into_iter().map(Ok))
        } else {
            Either::Left(once(Err(Error::from(ReadError::NColumns {
                at_row_index: row_index,
                expected: n_columns,
                actual: row_vec.len(),
            }))))
        }),
    });
    let array1_result: Result<Array1<A>, _> = values.collect();
    array1_result.and_then(|array1| {
        let array1_len = array1.len();
        let actual_n_rows = array1_len / n_columns;

        if actual_n_rows == n_rows {
            Ok(array1.into_shape(shape).unwrap_or_else(|_| {
                panic!(
                    "Reshaping from an Array1 of length {:?} to an Array2 of size {:?} failed",
                    array1_len, shape
                )
            }))
        } else {
            Err(Error::from(ReadError::NRows {
                expected: n_rows,
                actual: actual_n_rows,
            }))
        }
    })
}

/// Write this ndarray into CSV format
pub fn write<A>(array: &Array2<A>, writer: &mut Writer<impl Write>) -> Result<(), csv::Error>
where
    A: serde::Serialize,
{
    for row in array.outer_iter() {
        writer.serialize(row.as_slice())?;
    }
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::ReadError::*;
    use super::*;
    use csv::ReaderBuilder;
    use csv::WriterBuilder;
    use std::io::Cursor;

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
        let readed: Result<Array2<i8>, _> = read((2, 3), &mut in_memory_reader("1,2,3\n4,x,6\n"));
        readed.unwrap_err().downcast_ref::<csv::Error>().unwrap();
    }

    #[test]
    fn test_read_too_few_rows() {
        let readed: Result<Array2<i8>, _> = read((3, 3), &mut test_reader());
        assert_matches! {
            readed.unwrap_err().downcast_ref().unwrap(),
            NRows { expected: 3, actual: 2 }
        }
    }

    #[test]
    fn test_read_too_many_rows() {
        let readed: Result<Array2<i8>, _> = read((1, 3), &mut test_reader());
        assert_matches! {
            readed.unwrap_err().downcast_ref().unwrap(),
            NRows { expected: 1, actual: 2 }
        }
    }

    #[test]
    fn test_read_too_few_columns() {
        let readed: Result<Array2<i8>, _> = read((2, 4), &mut test_reader());
        assert_matches! {
            readed.unwrap_err().downcast_ref().unwrap(),
            NColumns { at_row_index: 0, expected: 4, actual: 3 }
        }
    }

    #[test]
    fn test_read_too_many_columns() {
        let readed: Result<Array2<i8>, _> = read((2, 2), &mut test_reader());
        assert_matches! {
            readed.unwrap_err().downcast_ref().unwrap(),
            NColumns { at_row_index: 0, expected: 2, actual: 3 }
        }
    }

    #[test]
    fn test_write_ok() {
        let mut writer = WriterBuilder::new().has_headers(false).from_writer(vec![]);

        assert_matches! {
            write(&array![[1, 2, 3], [4, 5, 6]], &mut writer),
            Ok(())
        }
        assert_eq!(
            writer.into_inner().expect("flush failed"),
            b"1,2,3\n4,5,6\n"
        );
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
