SELECT * FROM sp500_stocks LIMIT 10;

--Standardize Column Names
sql 
Copy code
ALTER TABLE sp500_stocks
RENAME COLUMN 'Adj Close' TO 'Adj_Close';
--

--Remove Duplicates
sql
Copy code
DELETE FROM sp500_stocks
WHERE rowid NOT IN (
    SELECT MIN(rowid)
    FROM sp500_stocks
    GROUP BY Date, Symbol
);
-- Handle Missing Values



UPDATE sp500_stocks
SET Volume = (
    SELECT AVG(Volume)
    FROM sp500_stocks
    WHERE Volume IS NOT NULL
)
WHERE Volume IS NULL;



DELETE FROM sp500_stocks
WHERE Date IS NULL OR Close IS NULL;
--Fix Data Types


ALTER TABLE sp500_stocks
ALTER COLUMN Date TYPE DATE USING TO_DATE(Date, 'YYYY-MM-DD');

-- Create New Features

ALTER TABLE sp500_stocks ADD COLUMN Daily_Return FLOAT;

UPDATE sp500_stocks a
SET Daily_Return = (a.Close - b.Close) / b.Close
FROM sp500_stocks b
WHERE a.Symbol = b.Symbol AND a.Date = b.Date + INTERVAL '1 day';
--Check for Outliers


SELECT Symbol, Date, Volume
FROM sp500_stocks
WHERE Volume > (SELECT AVG(Volume) + 3 * STDDEV(Volume) FROM sp500_stocks);