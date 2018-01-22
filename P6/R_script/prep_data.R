
library(dplyr)

# Data Loading
flights <- read.csv('2008.csv', header = T, 
                    sep = ',', check.names = F)

# Select columns
flights <- flights[, c("Year", "Month", "DayofMonth",
                                      "ArrDelay", "DepDelay")]
# Rename columns
names(flights) <- c("year", "month", "dayofmonth",
                              "arrdelay", "depdelay")

# Average days
flights_mean <- flights %>%
  group_by(year, month, dayofmonth) %>%
  summarise(arrdelay_mean = mean(arrdelay, na.rm = TRUE),
            depdelay_mean = mean(depdelay, na.rm = TRUE),
            n = n())

# Create m/d/Y column
flights_mean$date <- with(flights_mean, paste(month, dayofmonth, year, sep = "/"))

flights_mean$date <- as.Date(flights_mean$date, "%m/%d/%Y")

# Output the file
write.csv(flights_mean, file = "2008_prep2.csv", row.names = FALSE)


# plot prototypes
ggplot(data = flights_mean, aes(x = date)) +
  geom_line(aes(y = depdelay_mean))

ggplot(data = flights_mean, aes(x = date)) +
  geom_col(aes(y = n))

ggplot(data = flights_mean, aes(x = date)) +
  geom_col(aes(y = n)) +
  geom_line(aes(y = depdelay_mean), color = "red")

ggplot(data = flights_mean, aes(x = date)) +
  geom_col(aes(y = n / 1000)) +
  geom_line(aes(y = depdelay_mean), color = "red")

ggplot(data = flights_mean, aes(x = date)) +
  geom_line(aes(y = n / 1000)) +
  geom_line(aes(y = depdelay_mean), color = "red")

