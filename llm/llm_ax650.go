package llm

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/ml"
)

// AX650 backend configuration
const (
	AX650BackendURL = "http://localhost:5002"
)

// ax650Server implements LlamaServer interface for AX650/LLM8850 hardware
type ax650Server struct {
	llmServer
	backendURL string
	client     *http.Client
}

// NewAX650Server creates a new AX650 backend server instance
func NewAX650Server(modelPath string, opts api.Options) (LlamaServer, error) {
	backendURL := os.Getenv("AX650_BACKEND_URL")
	if backendURL == "" {
		backendURL = AX650BackendURL
	}

	server := &ax650Server{
		llmServer: llmServer{
			modelPath: modelPath,
			options:   opts,
			done:      make(chan error, 1),
		},
		backendURL: backendURL,
		client: &http.Client{
			Timeout: 300 * time.Second,
		},
	}

	return server, nil
}

// Load initializes the model on AX650 backend
func (s *ax650Server) Load(ctx context.Context, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool) ([]ml.DeviceID, error) {
	slog.Info("Loading model on AX650 backend", "model", s.modelPath, "backend", s.backendURL)

	// Call backend /load endpoint
	payload := map[string]string{
		"model_path": s.modelPath,
	}
	jsonData, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal load request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", s.backendURL+"/load", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create load request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := s.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to load model on AX650: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("AX650 backend returned error: %s", string(body))
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode load response: %w", err)
	}

	slog.Info("Model loaded on AX650", "result", result)

	// Return a dummy device ID (AX650 NPU)
	return []ml.DeviceID{{ID: "0", Library: "npu"}}, nil
}

// Ping checks if the AX650 backend is responsive
func (s *ax650Server) Ping(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, "GET", s.backendURL+"/health", nil)
	if err != nil {
		return err
	}

	resp, err := s.client.Do(req)
	if err != nil {
		return fmt.Errorf("AX650 backend not responding: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("AX650 backend health check failed: status %d", resp.StatusCode)
	}

	return nil
}

// WaitUntilRunning waits for the backend to be ready
func (s *ax650Server) WaitUntilRunning(ctx context.Context) error {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	timeout := time.After(30 * time.Second)

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-timeout:
			return errors.New("timeout waiting for AX650 backend")
		case <-ticker.C:
			if err := s.Ping(ctx); err == nil {
				return nil
			}
		}
	}
}

// Completion generates text completion using AX650 backend
func (s *ax650Server) Completion(ctx context.Context, req CompletionRequest, fn func(CompletionResponse)) error {
	slog.Debug("AX650 completion request", "prompt", req.Prompt[:min(50, len(req.Prompt))])

	// Prepare request for AX650 backend
	payload := map[string]interface{}{
		"prompt":      req.Prompt,
		"max_tokens":  cmp.Or(req.Options.NumPredict, 128),
		"temperature": cmp.Or(req.Options.Temperature, 0.8),
		"top_p":       cmp.Or(req.Options.TopP, 0.9),
		"top_k":       cmp.Or(req.Options.TopK, 40),
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal completion request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", s.backendURL+"/generate", bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("failed to create generate request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := s.client.Do(httpReq)
	if err != nil {
		return fmt.Errorf("AX650 generation failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("AX650 backend error: %s", string(body))
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return fmt.Errorf("failed to decode response: %w", err)
	}

	text, ok := result["text"].(string)
	if !ok {
		return errors.New("invalid response format from AX650 backend")
	}

	// Send completion response
	fn(CompletionResponse{
		Content: text,
		Done:    true,
	})

	return nil
}

// Embedding returns embeddings for the given input (not implemented for AX650)
func (s *ax650Server) Embedding(ctx context.Context, input string) ([]float32, error) {
	return nil, errors.New("embeddings not supported on AX650 backend")
}

// Tokenize converts text to tokens (delegated to backend if available)
func (s *ax650Server) Tokenize(ctx context.Context, content string) ([]int, error) {
	return nil, errors.New("tokenization not directly exposed by AX650 backend")
}

// Detokenize converts tokens to text (delegated to backend if available)
func (s *ax650Server) Detokenize(ctx context.Context, tokens []int) (string, error) {
	return "", errors.New("detokenization not directly exposed by AX650 backend")
}

// Close shuts down the AX650 backend connection
func (s *ax650Server) Close() error {
	slog.Info("Closing AX650 backend connection")
	close(s.done)
	return nil
}

// VRAMSize returns total VRAM (N/A for AX650, return 0)
func (s *ax650Server) VRAMSize() uint64 {
	return 0
}

// TotalSize returns total model size
func (s *ax650Server) TotalSize() uint64 {
	return 0
}

// VRAMByGPU returns VRAM for specific GPU (N/A for AX650)
func (s *ax650Server) VRAMByGPU(id ml.DeviceID) uint64 {
	return 0
}

// Pid returns the process ID (N/A for remote backend)
func (s *ax650Server) Pid() int {
	return 0
}

// GetPort returns the port number (N/A for remote backend)
func (s *ax650Server) GetPort() int {
	return 0
}

// GetDeviceInfos returns device information
func (s *ax650Server) GetDeviceInfos(ctx context.Context) []ml.DeviceInfo {
	return []ml.DeviceInfo{
		{
			DeviceID:    ml.DeviceID{ID: "0", Library: "npu"},
			Name:        "AX650/LLM8850 NPU",
			Description: "AXERA AX650/LLM8850 NPU",
		},
	}
}

// HasExited checks if the backend has exited
func (s *ax650Server) HasExited() bool {
	select {
	case <-s.done:
		return true
	default:
		return false
	}
}

// ModelPath returns the model path
func (s *ax650Server) ModelPath() string {
	return s.modelPath
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
