# Databricks notebook source
# MAGIC %sql
# MAGIC -- Run once in a Databricks SQL editor / notebook before launching the app
# MAGIC CREATE TABLE IF NOT EXISTS dev_digital_engineering_services.hackathon.gold.user_feedback (
# MAGIC   feedback_id     STRING,
# MAGIC   ts              STRING,
# MAGIC   user_email      STRING,
# MAGIC   query           STRING,
# MAGIC   answer          STRING,
# MAGIC   model_used      STRING,
# MAGIC   rating          STRING,        -- 'up' | 'down'
# MAGIC   comment         STRING,
# MAGIC   n_citations     INT,
# MAGIC   citation_ids    STRING
# MAGIC ) USING DELTA;
# MAGIC