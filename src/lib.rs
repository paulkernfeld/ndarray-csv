//! Easily read and write homogeneous CSV data to and from 2D ndarrays.
//!
//! ```rust
//! extern crate csv;
//! extern crate ndarray;
//! extern crate ndarray_csv;
//!
//! use csv::{ReaderBuilder, WriterBuilder};
//! use ndarray::{array, Array2};
//! use ndarray_csv::{Array2Reader, Array2Writer};
//! use std::error::Error;
//! use std::fs::File;
//!
//! fn main() -> Result<(), Box<dyn Error>> {
//!     // Our 2x3 test array
//!     let array = array![[1, 2, 3], [4, 5, 6]];
//!
//!     // Write the array into the file.
//!     {
//!         let file = File::create("test.csv")?;
//!         let mut writer = WriterBuilder::new().has_headers(false).from_writer(file);
//!         writer.serialize_array2(&array)?;
//!     }
//!
//!     // Read an array back from the file
//!     let file = File::open("test.csv")?;
//!     let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
//!     let array_read: Array2<u64> = reader.deserialize_array2((2, 3))?;
//!
//!     // Ensure that we got the original array back
//!     assert_eq!(array_read, array);
//!     Ok(())
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
#[cfg(test)]
#[macro_use]
extern crate matches;
#[cfg_attr(test, macro_use(array))]
extern crate ndarray;
extern crate serde;

use csv::{Reader, Writer};
use either::Either;
use ndarray::iter::Iter;
use ndarray::{Array1, Array2, Dim};
use serde::de::DeserializeOwned;
use serde::{Serialize, Serializer};
use std::cell::Cell;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::io::{Read, Write};
use std::iter::once;

/// An extension trait; this is implemented by `&mut csv::Reader`
pub trait Array2Reader {
    /// Read CSV data into a new ndarray with the given shape
    fn deserialize_array2<A: DeserializeOwned>(
        self,
        shape: (usize, usize),
    ) -> Result<Array2<A>, ReadError>;

    fn deserialize_array2_dynamic<A: DeserializeOwned>(self) -> Result<Array2<A>, ReadError>;
}

#[derive(Debug)]
pub enum ReadError {
    Csv(csv::Error),
    NRows {
        expected: usize,
        actual: usize,
    },
    NColumns {
        at_row_index: usize,
        expected: usize,
        actual: usize,
    },
}

impl Display for ReadError {
    fn fmt(&self, f: &mut Formatter) -> Result<(), std::fmt::Error> {
        match self {
            ReadError::Csv(csv_error) => csv_error.fmt(f),
            ReadError::NRows { expected, actual } => {
                write!(f, "Expected {} rows but got {} rows", expected, actual)
            }
            ReadError::NColumns {
                at_row_index,
                expected,
                actual,
            } => write!(
                f,
                "On row {}, expected {} columns but got {} columns",
                at_row_index, expected, actual
            ),
        }
    }
}

impl Error for ReadError {}

impl<'a, R: Read> Array2Reader for &'a mut Reader<R> {
    fn deserialize_array2<A: DeserializeOwned>(
        self,
        shape: (usize, usize),
    ) -> Result<Array2<A>, ReadError> {
        let (n_rows, n_columns) = shape;

        let rows = self.deserialize::<Vec<A>>();
        let values = rows.enumerate().flat_map(|(row_index, row)| match row {
            Err(e) => Either::Left(once(Err(ReadError::Csv(e)))),
            Ok(row_vec) => Either::Right(if row_vec.len() == n_columns {
                Either::Right(row_vec.into_iter().map(Ok))
            } else {
                Either::Left(once(Err(ReadError::NColumns {
                    at_row_index: row_index,
                    expected: n_columns,
                    actual: row_vec.len(),
                })))
            }),
        });
        let array1_result: Result<Array1<A>, _> = values.collect();
        array1_result.and_then(|array1| {
            let array1_len = array1.len();
            #[allow(deprecated)]
            array1.into_shape(shape).map_err(|_| ReadError::NRows {
                expected: n_rows,
                actual: array1_len / n_columns,
            })
        })
    }

    fn deserialize_array2_dynamic<A: DeserializeOwned>(self) -> Result<Array2<A>, ReadError> {
        let mut row_count = 0;
        let mut last_columns = None;

        let rows = self.deserialize::<Vec<A>>();
        let values = rows.enumerate().flat_map(|(row_index, row)| {
            row_count += 1;
            match row {
                Err(e) => Either::Left(once(Err(ReadError::Csv(e)))),
                Ok(row_vec) => {
                    if let Some(last_columns) = last_columns {
                        if last_columns != row_vec.len() {
                            return Either::Right(Either::Left(once(Err(ReadError::NColumns {
                                at_row_index: row_index,
                                expected: last_columns,
                                actual: row_vec.len(),
                            }))));
                        }
                    };
                    last_columns = Some(row_vec.len());
                    Either::Right(Either::Right(row_vec.into_iter().map(Ok)))
                }
            }
        });
        let array1_result: Result<Array1<A>, _> = values.collect();
        array1_result.map(|array1| {
            #[allow(deprecated)]
            array1
                .into_shape((row_count, last_columns.unwrap_or(0)))
                .unwrap()
        })
    }
}

/// An extension trait; this is implemented by `&mut csv::Writer`
pub trait Array2Writer {
    /// Write this ndarray into CSV format
    fn serialize_array2<A: Serialize>(self, array: &Array2<A>) -> Result<(), csv::Error>;
}

impl<'a, W: Write> Array2Writer for &'a mut Writer<W> {
    fn serialize_array2<A: Serialize>(self, array: &Array2<A>) -> Result<(), csv::Error> {
        /// This wraps the iterator for a row so that we can implement Serialize.
        ///
        /// Serialize is not implemented for iterators: https://github.com/serde-rs/serde/issues/571
        ///
        /// This solution from Hyeonu wraps the iterator:
        /// https://users.rust-lang.org/t/how-to-serialize-an-iterator-to-json/59272/3
        struct Row1DIter<'b, B>(Cell<Option<Iter<'b, B, Dim<[usize; 1]>>>>);

        impl<'b, B> Serialize for Row1DIter<'b, B>
        where
            B: Serialize,
        {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                serializer.collect_seq(self.0.take().unwrap())
            }
        }

        for row in array.outer_iter() {
            self.serialize(Row1DIter(Cell::new(Some(row.iter()))))?;
        }
        self.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::ReadError::*;
    use super::*;
    use csv::{Reader, ReaderBuilder, WriterBuilder};
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
        let actual: Array2<f64> = test_reader().deserialize_array2((2, 3)).unwrap();
        let expected = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_read_integer() {
        let actual: Array2<u64> = test_reader().deserialize_array2((2, 3)).unwrap();
        let expected = array![[1, 2, 3], [4, 5, 6]];
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_read_dynamic() {
        let actual: Array2<u64> = test_reader().deserialize_array2_dynamic().unwrap();
        let expected = array![[1, 2, 3], [4, 5, 6]];
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_read_csv_error() {
        in_memory_reader("1,2,3\n4,x,6\n")
            .deserialize_array2::<i8>((2, 3))
            .unwrap_err();
    }

    #[test]
    fn test_read_too_few_rows() {
        assert_matches! {
            test_reader().deserialize_array2::<i8>((3, 3)).unwrap_err(),
            NRows { expected: 3, actual: 2 }
        }
    }

    #[test]
    fn test_read_too_many_rows() {
        assert_matches! {
            test_reader().deserialize_array2::<i8>((1, 3)).unwrap_err(),
            NRows { expected: 1, actual: 2 }
        }
    }

    #[test]
    fn test_read_too_few_columns() {
        assert_matches! {
            test_reader().deserialize_array2::<i8>((2, 4)).unwrap_err(),
            NColumns { at_row_index: 0, expected: 4, actual: 3 }
        }
    }

    #[test]
    fn test_read_too_many_columns() {
        assert_matches! {
            test_reader().deserialize_array2::<i8>((2, 2)).unwrap_err(),
            NColumns { at_row_index: 0, expected: 2, actual: 3 }
        }
    }

    #[test]
    fn test_write_ok() {
        let mut writer = WriterBuilder::new().has_headers(false).from_writer(vec![]);

        assert_matches! {
            writer.serialize_array2(&array![[1, 2, 3], [4, 5, 6]]),
            Ok(())
        }
        assert_eq!(
            writer.into_inner().expect("flush failed"),
            b"1,2,3\n4,5,6\n"
        );
    }

    #[test]
    fn test_write_transposed() {
        let mut writer = WriterBuilder::new().has_headers(false).from_writer(vec![]);

        assert_matches! {
            writer.serialize_array2(&array![[1, 4], [2, 5], [3, 6]].t().to_owned()),
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
            writer.serialize_array2(&array![[1, 2, 3], [4, 5, 6]]),
            Err(_)
        }
    }
}
