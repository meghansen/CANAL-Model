# =============================================================================
# analyze_outputs.R
# =============================================================================
# Analysis and visualisation functions for Amsterdam influenza ABM outputs.
#
# Functions
# ---------
#   load_sim_data()            — load and type-correct a simulation CSV
#   load_wijk_mapping()        — load wijk index-to-name mapping (0-indexed)
#   parse_rate()               — parse rate strings like "~0.1-0.2%" to numeric
#   load_rate_lookup()         — load IHR/IFR table and parse rates
#   get_wijk_full_stats()      — per-wijk attack rates, hospitalisations, deaths
#   plot_age_attack_rates()    — bar chart of attack rate by 10-year age bin
#   plot_setting_by_age()      — stacked bar of infection setting by age group
#   plot_wijk_attack_rate()    — scatter: household size vs attack rate
#   plot_wijk_outcomes()       — side-by-side scatter: median age vs hosp/death
#
# Typical usage
# -------------
#   source("analyze_outputs.R")
#
#   data_csv  <- "validation_outputs/data_beta0.1_rec0.05_rep1.csv"
#   wijk_csv  <- "validation_outputs/wijk_idx_to_naam.csv"
#   rates_csv <- "data/age_rates.csv"
#
#   stats <- get_wijk_full_stats(data_csv, wijk_csv, rates_csv)
#
#   plot_age_attack_rates(data_csv)
#   plot_setting_by_age(data_csv)
#   plot_wijk_attack_rate(stats)
#   plot_wijk_outcomes(stats)
# =============================================================================

library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
library(ggpubr)
library(scales)


# -----------------------------------------------------------------------------
# Data loading helpers
# -----------------------------------------------------------------------------

#' Load and type-correct a simulation output CSV.
#'
#' Reads the per-agent CSV produced by run_parameter_set.py and coerces
#' columns to the correct types (Python sometimes writes integers as floats).
#'
#' @param data_path Path to the data_beta{b}_rec{r}_rep{k}.csv file.
#' @return A data frame with one row per agent.
load_sim_data <- function(data_path) {
  df <- read.csv(data_path, na.strings = c("", "NA", "NaN", "nan"))
  df$wijk         <- as.integer(df$wijk)
  df$exposure_day <- as.numeric(df$exposure_day)
  df$age          <- as.integer(df$age)
  df$household    <- as.integer(df$household)
  df$swstatus     <- as.integer(df$swstatus)
  return(df)
}


#' Load the wijk index-to-name mapping and add a 0-based integer index column.
#'
#' @param mapping_path Path to wijk_idx_to_naam.csv.
#' @return A data frame with columns: wijk (int, 0-based), name (chr).
load_wijk_mapping <- function(mapping_path) {
  read.csv(mapping_path) %>%
    mutate(wijk = row_number() - 1L) %>%
    rename(name = wijk_name)
}


#' Parse a rate string to a plain numeric proportion.
#'
#' Handles formats such as "0.5%", "~0.1-0.2%", "<1%", "0.3".
#' Ranges (e.g. "0.1-0.2") are averaged. The % symbol is removed and the
#' result is divided by 100.
#'
#' @param x Character string representing a rate.
#' @return Numeric proportion in [0, 1].
parse_rate <- function(x) {
  clean <- str_replace_all(as.character(x), "[~%< ]", "")
  if (str_detect(clean, "-")) {
    vals <- suppressWarnings(as.numeric(str_split(clean, "-", simplify = TRUE)))
    return(mean(vals, na.rm = TRUE) / 100)
  }
  return(suppressWarnings(as.numeric(clean)) / 100)
}


#' Load the age-specific IHR/IFR rates table and parse rate strings.
#'
#' Expects a CSV with at least:
#'   Age.Group                      — age group label matching the simulation bins
#'   SARS.CoV.2.IHR....of.infected. — infection-hospitalisation rate string
#'   SARS.CoV.2.IFR....of.infected. — infection-fatality rate string
#'
#' @param rates_path Path to the age rates CSV.
#' @return A data frame with columns: Age.Group, ihr_num, ifr_num.
load_rate_lookup <- function(rates_path) {
  read.csv(rates_path) %>%
    mutate(
      ihr_num = sapply(SARS.CoV.2.IHR....of.infected., parse_rate),
      ifr_num = sapply(SARS.CoV.2.IFR....of.infected., parse_rate)
    ) %>%
    select(Age.Group, ihr_num, ifr_num)
}


# -----------------------------------------------------------------------------
# Per-wijk summary statistics
# -----------------------------------------------------------------------------

#' Compute per-wijk epidemic summary statistics including hospitalisations
#' and deaths.
#'
#' For each wijk, calculates:
#'   population         — total agents
#'   infections         — agents with exposure_day > 0
#'   attack_rate        — infections / population
#'   median_age         — median age of agents
#'   mean_hh_size       — mean household size
#'   est_hospitalizations — expected hospitalisations (infections * IHR by age group)
#'   hosp_prop          — est_hospitalizations / population
#'   est_deaths         — expected deaths (infections * IFR by age group)
#'   death_prop         — est_deaths / population
#'
#' Age bins used to match the rates CSV:
#'   <10, 10-19, 20-39, 40-49, 50-59, 60-69, 70-79, 80-84, >=85 years
#'
#' @param data_path    Path to the per-agent simulation CSV.
#' @param mapping_path Path to wijk_idx_to_naam.csv.
#' @param rates_path   Path to the age-specific IHR/IFR CSV.
#' @return A data frame with one row per wijk.
get_wijk_full_stats <- function(data_path, mapping_path, rates_path) {

  df          <- load_sim_data(data_path)
  mapping_df  <- load_wijk_mapping(mapping_path)
  rate_lookup <- load_rate_lookup(rates_path)

  # Bin ages to match the rate table categories
  age_breaks <- c(0, 10, 20, 40, 50, 60, 70, 80, 85, Inf)
  age_labels  <- c("<10 yrs", "10-19 yrs", "20-39 yrs", "40-49 yrs",
                   "50-59 yrs", "60-69 yrs", "70-79 yrs", "80-84 yrs", ">=85 yrs")

  df <- df %>%
    mutate(age_group = cut(age, breaks = age_breaks, labels = age_labels,
                           right = FALSE, include.lowest = TRUE))

  # Expected hospitalisations and deaths per wijk
  outcome_calc <- df %>%
    filter(exposure_day > 0) %>%
    group_by(wijk, age_group) %>%
    summarise(infections_in_group = dplyr::n(), .groups = "drop") %>%
    left_join(rate_lookup, by = c("age_group" = "Age.Group")) %>%
    mutate(
      exp_hosp  = infections_in_group * ihr_num,
      exp_death = infections_in_group * ifr_num
    ) %>%
    group_by(wijk) %>%
    summarise(
      est_hospitalizations = sum(exp_hosp,  na.rm = TRUE),
      est_deaths           = sum(exp_death, na.rm = TRUE),
      .groups = "drop"
    )

  # Mean household size per wijk
  hh_stats <- df %>%
    group_by(wijk, household) %>%
    summarise(hh_size = dplyr::n(), .groups = "drop") %>%
    group_by(wijk) %>%
    summarise(mean_hh_size = mean(hh_size), .groups = "drop")

  # Core demographic and attack rate summary
  summary_stats <- df %>%
    group_by(wijk) %>%
    summarise(
      population  = dplyr::n(),
      median_age  = median(age, na.rm = TRUE),
      infections  = sum(exposure_day > 0, na.rm = TRUE),
      attack_rate = infections / population,
      .groups = "drop"
    )

  # Combine all components
  summary_stats %>%
    left_join(outcome_calc, by = "wijk") %>%
    left_join(hh_stats,     by = "wijk") %>%
    left_join(mapping_df,   by = "wijk") %>%
    mutate(
      est_hospitalizations = replace_na(est_hospitalizations, 0),
      est_deaths           = replace_na(est_deaths, 0),
      hosp_prop            = est_hospitalizations / population,
      death_prop           = est_deaths / population
    ) %>%
    select(wijk, name, population, infections, attack_rate,
           median_age, mean_hh_size,
           est_hospitalizations, hosp_prop,
           est_deaths, death_prop)
}


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------

#' Bar chart of attack rate by 10-year age group.
#'
#' @param data_path Path to the per-agent simulation CSV.
#' @return A ggplot object.
plot_age_attack_rates <- function(data_path) {
  df <- load_sim_data(data_path) %>%
    mutate(age_bin = cut(age,
                         breaks = seq(0, 110, by = 10),
                         labels = paste0(seq(0, 100, by = 10), "-",
                                         seq(10, 110, by = 10)),
                         right = FALSE, include.lowest = TRUE))

  age_stats <- df %>%
    group_by(age_bin) %>%
    summarise(
      population   = dplyr::n(),
      infections  = sum(exposure_day > 0, na.rm = TRUE),
      attack_rate = infections / population,
      .groups = "drop"
    ) %>%
    filter(!is.na(age_bin))

  ggplot(age_stats, aes(x = age_bin, y = attack_rate)) +
    geom_col(fill = "steelblue", colour = "white") +
    geom_text(aes(label = percent(attack_rate, accuracy = 0.1)),
              vjust = -0.5, size = 3) +
    scale_y_continuous(labels = percent,
                       limits = c(0, max(age_stats$attack_rate) * 1.2)) +
    labs(
      title    = "Attack Rate by 10-Year Age Group",
      x        = "Age Group",
      y        = "Attack Rate"
    ) +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}


#' Stacked bar chart of infection setting by broad age group.
#'
#' Shows the percentage of infections occurring in each setting
#' (household, school, workplace, community) for three age groups:
#' <18, 18-64, and 65+ years. Seeded infections (setting = 5) are excluded.
#'
#' @param data_path Path to the per-agent simulation CSV.
#' @return A ggplot object.
plot_setting_by_age <- function(data_path) {
  df <- load_sim_data(data_path)

  df_plot <- df %>%
    filter(exposure_setting %in% 1:4) %>%
    mutate(
      age_group = cut(age,
                      breaks = c(0, 18, 64, Inf),
                      labels = c("<18 years", "18-64 years", "65+ years"),
                      right  = TRUE),
      setting = factor(exposure_setting,
                       levels = 1:4,
                       labels = c("Household", "School", "Workplace", "Community"))
    ) %>%
    filter(!is.na(age_group)) %>%
    count(age_group, setting) %>%
    group_by(age_group) %>%
    mutate(pct = n / sum(n) * 100) %>%
    ungroup()

  ggplot(df_plot, aes(x = age_group, y = pct, fill = setting)) +
    geom_col(position = "stack") +
    geom_text(aes(label = paste0(round(pct), "%")),
              position = position_stack(vjust = 0.5),
              size = 3.5, colour = "white", fontface = "bold") +
    scale_fill_manual(values = c(
      "Household"  = "#E07B54",
      "School"     = "#7B9E87",
      "Workplace"  = "#C9B1BD",
      "Community"  = "#4A7B9D"
    )) +
    scale_y_continuous(labels = percent_format(scale = 1)) +
    labs(
      title    = "Infection Setting by Age Group",
      subtitle = "Proportion of infections occurring in each setting",
      x        = "Age Group",
      y        = "Percentage of Infections",
      fill     = "Setting"
    ) +
    theme_bw() +
    theme(
      plot.title      = element_text(face = "bold"),
      legend.position = "right"
    )
}


#' Scatter plot of mean household size vs attack rate by wijk.
#'
#' Points are coloured by median age. Wijken with fewer than
#' min_population residents are excluded to reduce noise from very
#' small neighbourhoods.
#'
#' @param wijk_stats  Data frame returned by get_wijk_full_stats().
#' @param min_population Minimum wijk population to include (default 1000).
#' @return A ggplot object.
plot_wijk_attack_rate <- function(wijk_stats, min_population = 1000) {
  wijk_stats %>%
    filter(population >= min_population) %>%
    ggplot(aes(x = mean_hh_size, y = attack_rate, colour = median_age)) +
    geom_point(size = 2) +
    stat_smooth(method = "lm", colour = "grey30", se = TRUE) +
    scale_colour_viridis_c(name = "Median age") +
    scale_y_continuous(labels = percent_format(accuracy = 1)) +
    labs(
      title = "Household Size vs Attack Rate by Wijk",
      x     = "Mean Household Size",
      y     = "Attack Rate"
    ) +
    theme_bw()
}


#' Side-by-side scatter plots of median age vs hospitalisation and death rate.
#'
#' Each point is one wijk. Wijken with fewer than min_population residents
#' are excluded.
#'
#' @param wijk_stats     Data frame returned by get_wijk_full_stats().
#' @param min_population Minimum wijk population to include (default 1000).
#' @return A combined ggplot object (via ggpubr::ggarrange).
plot_wijk_outcomes <- function(wijk_stats, min_population = 1000) {
  df <- wijk_stats %>% filter(population >= min_population)

  p_hosp <- ggplot(df, aes(x = median_age, y = hosp_prop)) +
    geom_point(size = 2, colour = "#4A7B9D") +
    stat_smooth(method = "lm", colour = "grey30", se = TRUE) +
    scale_y_continuous(labels = percent_format(accuracy = 0.1)) +
    labs(
      title = "Hospitalisations by Wijk",
      x     = "Median Age",
      y     = "Proportion of Population Hospitalised"
    ) +
    theme_bw()

  p_death <- ggplot(df, aes(x = median_age, y = death_prop)) +
    geom_point(size = 2, colour = "#E07B54") +
    stat_smooth(method = "lm", colour = "grey30", se = TRUE) +
    scale_y_continuous(labels = percent_format(accuracy = 0.01)) +
    labs(
      title = "Deaths by Wijk",
      x     = "Median Age",
      y     = "Proportion of Population Dead"
    ) +
    theme_bw()

  ggarrange(p_hosp, p_death, ncol = 2)
}


# =============================================================================
# Example usage (comment out or wrap in if (FALSE) { } for sourcing) 
# =============================================================================

if (TRUE) {
  data_csv  <- "validation_outputs/sim_beta0.2_rec0.15_output.csv"
  wijk_csv  <- "validation_outputs/wijk_idx_to_naam.csv"
  rates_csv <- "SARS_hosp_rates.csv"

  # Compute per-wijk statistics including hospitalisations and deaths
  stats <- get_wijk_full_stats(data_csv, wijk_csv, rates_csv)
  print(head(stats))

  # Attack rate by age group
  plot_age_attack_rates(data_csv)

  # Infection setting by age group
  plot_setting_by_age(data_csv)

  # Wijk-level plots (requires stats from get_wijk_full_stats)
  plot_wijk_attack_rate(stats)
  plot_wijk_outcomes(stats)
}
