-- Neon PostgreSQL Schema Setup for Textbook RAG Chatbot
-- Run this script in your Neon database console

-- Chat sessions table
CREATE TABLE IF NOT EXISTS chat_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    message_count INT DEFAULT 0,
    client_hash VARCHAR(64)
);

-- Rate limits table
CREATE TABLE IF NOT EXISTS rate_limits (
    client_hash VARCHAR(64) PRIMARY KEY,
    request_count INT DEFAULT 0,
    window_start TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_sessions_created ON chat_sessions(created_at);
CREATE INDEX IF NOT EXISTS idx_rate_limits_window ON rate_limits(window_start);

-- Clean up old rate limit entries (run periodically)
-- DELETE FROM rate_limits WHERE window_start < NOW() - INTERVAL '2 hours';
