# Obscura Browser Automation Setup

Obscura is a lightweight, fast, anti-detection headless browser for AI agents.

## Installation

### Linux (x86_64)

```bash
# Download and install
curl -LO https://github.com/h4ckf0r0day/obscura/releases/latest/download/obscura-x86_64-linux.tar.gz
tar xzf obscura-x86_64-linux.tar.gz
sudo mv obscura /usr/local/bin/
sudo chmod +x /usr/local/bin/obscura

# Verify installation
obscura --version
```

### macOS (Apple Silicon)

```bash
curl -LO https://github.com/h4ckf0r0day/obscura/releases/latest/download/obscura-aarch64-macos.tar.gz
tar xzf obscura-aarch64-macos.tar.gz
sudo mv obscura /usr/local/bin/
sudo chmod +x /usr/local/bin/obscura
```

### macOS (Intel)

```bash
curl -LO https://github.com/h4ckf0r0day/obscura/releases/latest/download/obscura-x86_64-macos.tar.gz
tar xzf obscura-x86_64-macos.tar.gz
sudo mv obscura /usr/local/bin/
sudo chmod +x /usr/local/bin/obscura
```

### Windows

Download the `.zip` from the [releases page](https://github.com/h4ckf0r0day/obscura/releases) and extract it manually. Add the obscura binary to your PATH.

## Available Tools

### obscura_fetch

Fetch and render a webpage using Obscura headless browser.

**Parameters:**
- `url` (required): The URL to fetch
- `dump`: Output format - 'html', 'text', or 'links' (default: 'html')
- `eval_js`: JavaScript expression to evaluate on the page
- `wait_until`: When to consider page loaded - 'load', 'domcontentloaded', 'networkidle0' (default: 'load')
- `selector`: Optional CSS selector to wait for before returning
- `stealth`: Enable anti-detection mode (default: false)

**Example:**
```json
{
  "name": "obscura_fetch",
  "arguments": {
    "url": "https://example.com",
    "dump": "text",
    "wait_until": "networkidle0"
  }
}
```

### obscura_scrape

Scrape multiple URLs using Obscura. Note: Parallel scraping requires building the obscura-worker binary from source. Currently falls back to sequential fetching.

**Parameters:**
- `urls` (required): List of URLs to scrape
- `concurrency`: Number of parallel workers (default: 10) - currently limited to sequential
- `eval_js`: JavaScript expression to evaluate on each page
- `format`: Output format - 'json' or 'text' (default: 'json')

**Example:**
```json
{
  "name": "obscura_scrape",
  "arguments": {
    "urls": ["https://example.com", "https://example.org"],
    "concurrency": 5,
    "dump": "text",
    "format": "json"
  }
}
```

### obscura_serve

Start an Obscura CDP (Chrome DevTools Protocol) server for Puppeteer/Playwright integration.
The server is already running by default on port 9222.

**Parameters:**
- `port`: WebSocket port for CDP server (default: 9222)
- `stealth`: Enable anti-detection mode (default: false)
- `workers`: Number of parallel worker processes (default: 1)
- `proxy`: HTTP/SOCKS5 proxy URL

**Example:**
```json
{
  "name": "obscura_serve",
  "arguments": {
    "port": 9222,
    "stealth": false
  }
}
```

## Features

- **Lightweight**: ~30 MB memory usage (vs 200+ MB for Chrome)
- **Fast**: Page loads in ~85ms (vs ~500ms for Chrome)
- **Anti-detection**: Built-in stealth mode with fingerprint randomization
- **CDP Compatible**: Works with Puppeteer and Playwright
- **No Dependencies**: Single binary, no Chrome/Node.js required

## Usage Examples

### Fetch page content
```
User: Get the title of https://news.ycombinator.com
Agent: obscura_fetch(url="https://news.ycombinator.com", dump="text")
```

### Extract specific data
```
User: Get all links from example.com
Agent: obscura_fetch(url="https://example.com", dump="links")
```

### Evaluate JavaScript
```
User: Get the current temperature from a weather site
Agent: obscura_fetch(
  url="https://weather.example.com",
  eval_js="document.querySelector('.temperature').textContent"
)
```

### Scrape multiple pages
```
User: Scrape titles from these 5 news sites
Agent: obscura_scrape(
  urls=[...],
  concurrency=3,
  dump="text"
)
```

## Troubleshooting

### "Obscura binary not found"

Make sure Obscura is installed and in your PATH:
```bash
which obscura
obscura --version
```

### Connection issues

If you're behind a proxy, use the `proxy` parameter in `obscura_serve` or set environment variables:
```bash
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
```

## References

- [Obscura GitHub](https://github.com/h4ckf0r0day/obscura)
- [Obscura Releases](https://github.com/h4ckf0r0day/obscura/releases)
