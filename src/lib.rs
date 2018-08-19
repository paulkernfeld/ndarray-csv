//! Easily read homogeneous CSV data into a 2D ndarray.
//!
//! ```rust
//! extern crate csv;
//! extern crate ndarray_csv;
//!
//! use csv::ReaderBuilder;
//! use ndarray_csv::read;
//! use std::fs::File;
//!
//! fn main() {
//!     let file = File::open("test.csv").expect("opening test.csv failed");
//!     let reader = ReaderBuilder::new().has_headers(false).from_reader(file);
//!     let array = read::<f64, _>((2, 3), reader).expect("read failed");
//!     assert_eq!(array.dim(), (2, 3));
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

use csv::Reader;
use ndarray::Array2;
use std::io::Read;

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
pub fn read<A, R>(shape: (usize, usize), mut reader: Reader<R>) -> Result<Array2<A>, Error>
where
    R: Read,
    A: Copy,
    for<'de> A: serde::de::Deserialize<'de>,
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

#[cfg(test)]
mod tests {
    use super::Error::*;
    use super::*;
    use csv::ReaderBuilder;
    use std::io::Cursor;

    fn in_memory_reader(content: &'static str) -> Reader<impl Read> {
        ReaderBuilder::new()
            .has_headers(false)
            .from_reader(Cursor::new(content))
    }

    fn test_reader() -> Reader<impl Read> {
        in_memory_reader("1,2,3\n4,5,6")
    }

    #[test]
    fn test_read_float() {
        let actual = read((2, 3), test_reader()).unwrap();
        let expected = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_read_integer() {
        let actual = read((2, 3), test_reader()).unwrap();
        let expected = array![[1, 2, 3], [4, 5, 6]];
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_read_csv_error() {
        assert_matches! {
            read::<i8, _>((2, 3), in_memory_reader("1,2,3\n4,x,6")),
            Err(Csv(_))
        }
    }

    #[test]
    fn test_read_too_few_rows() {
        assert_matches! {
            read::<i8, _>((3, 3), test_reader()),
            Err(TooFewRows { expected: 3, actual: 2})
        }
    }

    #[test]
    fn test_read_too_many_rows() {
        assert_matches! {
            read::<i8, _>((1, 3), test_reader()),
            Err(TooManyRows { expected: 1 })
        }
    }

    #[test]
    fn test_read_too_few_columns() {
        assert_matches! {
            read::<i8, _>((2, 4), test_reader()),
            Err(TooFewColumns { at_row_index: 0, expected: 4, actual: 3 })
        }
    }

    #[test]
    fn test_read_too_many_columns() {
        assert_matches! {
            read::<i8, _>((2, 2), test_reader()),
            Err(TooManyColumns { at_row_index: 0, expected: 2 })
        }
    }
}
