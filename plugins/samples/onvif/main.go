package main

import (
    "encoding/json"
    "fmt"
    "os"
)

// Simple stub that reads JSON from stdin and echoes processed JSON to stdout.
func main() {
    var input map[string]interface{}
    dec := json.NewDecoder(os.Stdin)
    if err := dec.Decode(&input); err != nil {
        fmt.Fprintf(os.Stderr, "error decoding input: %v\n", err)
        os.Exit(1)
    }
    // Example compute-heavy placeholder: just echo with a field
    input["processed_by"] = "go_worker_onvif"
    enc := json.NewEncoder(os.Stdout)
    _ = enc.Encode(input)
}
