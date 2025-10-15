package main

import (
    "encoding/json"
    "fmt"
    "os"
)

func main() {
    var input map[string]interface{}
    dec := json.NewDecoder(os.Stdin)
    if err := dec.Decode(&input); err != nil {
        fmt.Fprintf(os.Stderr, "error decoding input: %v\n", err)
        os.Exit(1)
    }
    // Example: add a signature placeholder
    input["signed_by"] = "go_worker_notify"
    enc := json.NewEncoder(os.Stdout)
    _ = enc.Encode(input)
}
