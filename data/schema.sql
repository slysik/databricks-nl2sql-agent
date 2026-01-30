-- ============================================================================
-- NL2SQL Demo - Delta Tables Schema
-- ============================================================================
-- INTERVIEW NOTE: This creates sample Delta tables for the chatbot demo.
-- Run this in a Databricks notebook or SQL warehouse.
-- ============================================================================

-- Create catalog and schema
CREATE CATALOG IF NOT EXISTS ryder_chatbot;
USE CATALOG ryder_chatbot;

CREATE SCHEMA IF NOT EXISTS agent_accessible;
USE SCHEMA agent_accessible;

-- ============================================================================
-- Table 1: Vendors (Performance Tracking)
-- ============================================================================
CREATE TABLE IF NOT EXISTS vendors (
    vendor_id STRING COMMENT 'Unique vendor identifier',
    vendor_name STRING COMMENT 'Vendor company name',
    performance_score INT COMMENT 'Performance score (0-100)',
    reliability_pct DECIMAL(5,2) COMMENT 'Reliability percentage',
    contract_start_date DATE COMMENT 'Contract start date',
    region STRING COMMENT 'Operating region'
) USING DELTA
COMMENT 'Vendor performance and reliability metrics';

-- Insert sample data
INSERT INTO vendors VALUES
('V001', 'PackCo Inc', 92, 98.5, '2023-01-15', 'Southwest'),
('V002', 'BoxMasters', 87, 94.2, '2022-06-01', 'Midwest'),
('V003', 'Premium Pack', 95, 99.1, '2023-03-20', 'Southwest'),
('V004', 'QuickShip LLC', 78, 91.5, '2022-11-10', 'Northeast'),
('V005', 'Reliable Logistics', 89, 96.8, '2023-07-01', 'Southeast');

-- ============================================================================
-- Table 2: Routes (Fleet Operations)
-- ============================================================================
CREATE TABLE IF NOT EXISTS routes (
    route_id STRING COMMENT 'Unique route identifier',
    origin_hub STRING COMMENT 'Starting hub location',
    destination STRING COMMENT 'Destination city',
    distance_miles INT COMMENT 'Route distance in miles',
    avg_delivery_hours DECIMAL(4,1) COMMENT 'Average delivery time in hours',
    fuel_cost_per_trip DECIMAL(6,2) COMMENT 'Fuel cost per trip in USD'
) USING DELTA
COMMENT 'Fleet route information and metrics';

-- Insert sample data
INSERT INTO routes VALUES
('R001', 'Phoenix', 'Los Angeles', 370, 5.5, 125.50),
('R002', 'Phoenix', 'Las Vegas', 300, 4.5, 98.00),
('R003', 'Dallas', 'Houston', 240, 3.5, 82.00),
('R004', 'Chicago', 'Detroit', 280, 4.0, 95.50),
('R005', 'Phoenix', 'Tucson', 115, 2.0, 42.00);

-- ============================================================================
-- Table 3: Customer Questions (Support Analytics)
-- ============================================================================
CREATE TABLE IF NOT EXISTS customer_questions (
    question_id STRING COMMENT 'Unique question identifier',
    topic STRING COMMENT 'Question category/topic',
    question_text STRING COMMENT 'Customer question text',
    resolution_time_mins INT COMMENT 'Time to resolve in minutes',
    satisfaction_score INT COMMENT 'Customer satisfaction (1-5)',
    created_date DATE COMMENT 'Question submission date'
) USING DELTA
COMMENT 'Customer support questions and metrics';

-- Insert sample data
INSERT INTO customer_questions VALUES
('Q001', 'Billing', 'How do I update payment method?', 12, 5, '2024-01-15'),
('Q002', 'Tracking', 'Where is my shipment?', 8, 4, '2024-01-15'),
('Q003', 'Returns', 'How do I return an item?', 15, 4, '2024-01-16'),
('Q004', 'Billing', 'Invoice discrepancy', 25, 3, '2024-01-16'),
('Q005', 'Tracking', 'Delivery status update', 5, 5, '2024-01-17'),
('Q006', 'Technical', 'API integration help', 45, 4, '2024-01-17'),
('Q007', 'Billing', 'Request invoice copy', 10, 5, '2024-01-18');

-- ============================================================================
-- RBAC Configuration (Security)
-- ============================================================================
-- INTERVIEW NOTE: In production, create a service principal with read-only access.
-- Uncomment and modify these for your environment:

-- Create read-only service principal grant
-- GRANT USE CATALOG ON CATALOG ryder_chatbot TO `nl2sql-agent-sp`;
-- GRANT USE SCHEMA ON SCHEMA ryder_chatbot.agent_accessible TO `nl2sql-agent-sp`;
-- GRANT SELECT ON SCHEMA ryder_chatbot.agent_accessible TO `nl2sql-agent-sp`;

-- Verify tables were created
SHOW TABLES IN ryder_chatbot.agent_accessible;

-- Sample queries for verification
-- SELECT * FROM vendors WHERE reliability_pct > 95;
-- SELECT * FROM routes WHERE origin_hub = 'Phoenix';
-- SELECT topic, COUNT(*) as count FROM customer_questions GROUP BY topic;
