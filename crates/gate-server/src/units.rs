//! Human-readable durations ("250ms", "30s", "10m", "2h") and sizes
//! ("512MiB", "32GiB", "1GB") for configuration files.
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Span(pub Duration);

impl Span {
    pub fn millis(self) -> u64 {
        self.0.as_millis() as u64
    }

    pub fn secs(seconds: u64) -> Self {
        Self(Duration::from_secs(seconds))
    }
}

impl std::str::FromStr for Span {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let text = text.trim();
        let split = text
            .find(|character: char| !(character.is_ascii_digit() || character == '.'))
            .unwrap_or(text.len());
        let (number, unit) = text.split_at(split);
        let value: f64 = number
            .parse()
            .map_err(|_| format!("invalid duration {text:?}"))?;
        let seconds = match unit.trim() {
            "ms" => value / 1000.0,
            "" | "s" => value,
            "m" => value * 60.0,
            "h" => value * 3600.0,
            "d" => value * 86_400.0,
            other => return Err(format!("unknown duration unit {other:?} in {text:?}")),
        };
        if !seconds.is_finite() || seconds < 0.0 {
            return Err(format!("invalid duration {text:?}"));
        }
        Ok(Self(Duration::from_secs_f64(seconds)))
    }
}

impl std::fmt::Display for Span {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let millis = self.0.as_millis();
        if millis.is_multiple_of(3_600_000) && millis > 0 {
            write!(formatter, "{}h", millis / 3_600_000)
        } else if millis.is_multiple_of(60_000) && millis > 0 {
            write!(formatter, "{}m", millis / 60_000)
        } else if millis.is_multiple_of(1000) {
            write!(formatter, "{}s", millis / 1000)
        } else {
            write!(formatter, "{millis}ms")
        }
    }
}

impl<'de> Deserialize<'de> for Span {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            Seconds(u64),
            Text(String),
        }
        match Raw::deserialize(deserializer)? {
            Raw::Seconds(seconds) => Ok(Span::secs(seconds)),
            Raw::Text(text) => text.parse().map_err(serde::de::Error::custom),
        }
    }
}

impl Serialize for Span {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Size(pub u64);

impl std::str::FromStr for Size {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let text = text.trim();
        let split = text
            .find(|character: char| !character.is_ascii_digit())
            .unwrap_or(text.len());
        let (number, unit) = text.split_at(split);
        let value: u64 = number
            .parse()
            .map_err(|_| format!("invalid size {text:?}"))?;
        let factor: u64 = match unit.trim() {
            "" | "B" => 1,
            "KB" => 1_000,
            "MB" => 1_000_000,
            "GB" => 1_000_000_000,
            "TB" => 1_000_000_000_000,
            "KiB" => 1 << 10,
            "MiB" => 1 << 20,
            "GiB" => 1 << 30,
            "TiB" => 1 << 40,
            other => return Err(format!("unknown size unit {other:?} in {text:?}")),
        };
        value
            .checked_mul(factor)
            .map(Size)
            .ok_or_else(|| format!("size {text:?} is too large"))
    }
}

impl std::fmt::Display for Size {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (unit, factor) in [
            ("TiB", 1u64 << 40),
            ("GiB", 1 << 30),
            ("MiB", 1 << 20),
            ("KiB", 1 << 10),
        ] {
            if self.0 >= factor && self.0.is_multiple_of(factor) {
                return write!(formatter, "{}{unit}", self.0 / factor);
            }
        }
        write!(formatter, "{}B", self.0)
    }
}

impl<'de> Deserialize<'de> for Size {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            Bytes(u64),
            Text(String),
        }
        match Raw::deserialize(deserializer)? {
            Raw::Bytes(bytes) => Ok(Size(bytes)),
            Raw::Text(text) => text.parse().map_err(serde::de::Error::custom),
        }
    }
}

impl Serialize for Size {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durations_and_sizes_parse_and_print_back() {
        for (text, millis) in [
            ("250ms", 250),
            ("30s", 30_000),
            ("10m", 600_000),
            ("2h", 7_200_000),
            ("1.5s", 1_500),
        ] {
            let span: Span = text.parse().unwrap();
            assert_eq!(span.millis(), millis, "{text}");
        }
        assert_eq!(Span::secs(90).to_string(), "90s");
        assert_eq!("32GiB".parse::<Size>().unwrap().0, 32 << 30);
        assert_eq!("1GB".parse::<Size>().unwrap().0, 1_000_000_000);
        assert_eq!(Size(64 << 20).to_string(), "64MiB");
        assert!("5 parsecs".parse::<Span>().is_err());
        assert!("12XB".parse::<Size>().is_err());
    }
}
