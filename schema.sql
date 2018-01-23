station: three or four character site identifier
valid: timestamp of the observation
tmpf: Air Temperature in Fahrenheit, typically @ 2 meters
dwpf: Dew Point Temperature in Fahrenheit, typically @ 2 meters
relh: Relative Humidity in %
drct: Wind Direction in degrees from north
sknt: Wind Speed in knots
p01i: One hour precipitation for the period from the observation time to the time of the previous hourly precipitation reset. This varies slightly by site. Values are in inches. This value may or may not contain frozen precipitation melted by some device on the sensor or estimated by some other means. Unfortunately, we do not know of an authoritative database denoting which station has which sensor.
alti: Pressure altimeter in inches
mslp: Sea Level Pressure in millibar
vsby: Visibility in miles
gust: Wind Gust in knots
skyc1: Sky Level 1 Coverage
skyc2: Sky Level 2 Coverage
skyc3: Sky Level 3 Coverage
skyc4: Sky Level 4 Coverage
skyl1: Sky Level 1 Altitude in feet
skyl2: Sky Level 2 Altitude in feet
skyl3: Sky Level 3 Altitude in feet
skyl4: Sky Level 4 Altitude in feet
presentwx: Present Weather Codes (space seperated)
metar: unprocessed reported observation in METAR format

# Create table

CREATE TABLE metar(
  station VARCHAR(3),
  valid timestamp,
  tmpf VARCHAR(24),
  dwpf VARCHAR(24),
  relh VARCHAR(24),
  drct VARCHAR(24),
  sknt VARCHAR(24),
  p01i VARCHAR(24),
  alti VARCHAR(24),
  mslp VARCHAR(24),
  vsby VARCHAR(24),
  gust VARCHAR(24),
  skyc1 VARCHAR(24),
  skyc2 VARCHAR(24),
  skyc3 VARCHAR(24),
  skyc4 VARCHAR(24),
  skyl1 VARCHAR(24),
  skyl2 VARCHAR(24),
  skyl3 VARCHAR(24),
  skyl4 VARCHAR(24),
  presentwx VARCHAR(64),
  metar VARCHAR(128));

  CREATE TABLE flights(
    flight INT PRIMARY KEY,
    log TEXT);

# Chande data type

ALTER TABLE metar
ALTER COLUMN tmpf TYPE VARCHAR(24),
ALTER COLUMN dwpf TYPE VARCHAR(24),
ALTER COLUMN relh TYPE VARCHAR(24),
ALTER COLUMN p01i TYPE VARCHAR(24);

# update records

UPDATE metar
SET DATE = valid ::timestamp::date,
SET TIME = valid ::timestamp::time;

SELECT COUNT(tmpf)
FROM metar
WHERE tmpf = 'M'
GROUP By valid
