"""
live_view.py - Auto-refreshing browser view of a run's conversation.md.

Usage:
    python scripts/live_view.py <output/run_id | run_id | path/to/conversation.md> [port]

Opens a tiny local web server (default http://localhost:8000) that live-updates the
run's conversation as it progresses. READ-ONLY: it never touches the run.

Smart refresh: the page fetches new content in the background and updates it
INCREMENTALLY (append-only diff) - unchanged blocks are left in place and a block
containing a <video> is never replaced once rendered, so recorded demo videos never
flicker, reset their playback, or reshuffle while the page keeps updating. It also
STOPS refreshing automatically when the run finishes (no more jitter on a completed run).
"""
import sys
import os
import html
import http.server
import functools

POLL_MS = 3000
DEFAULT_PORT = 8000

try:
    import markdown as _markdown
except Exception:
    _markdown = None


def resolve_paths(arg):
    """Return (output_dir, md_path) from a run_id, an output dir, or a conversation.md path."""
    arg = arg.rstrip("/\\")
    if os.path.isfile(arg) and arg.endswith(".md"):
        return (os.path.dirname(arg) or "."), arg
    candidate = arg
    if not os.path.isdir(candidate):
        candidate = os.path.join("output", arg)
    return candidate, os.path.join(candidate, "conversation.md")


def render_body(md_path):
    if not os.path.isfile(md_path):
        return "<h2>Waiting for conversation.md&hellip;</h2><p>Looked for: <code>{}</code></p>".format(
            html.escape(md_path))
    with open(md_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    if _markdown is not None:
        return _markdown.markdown(text, extensions=["fenced_code", "tables", "nl2br"])
    return "<pre>{}</pre>".format(html.escape(text))


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ background:#0d1117; color:#c9d1d9; font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;
         max-width:920px; margin:0 auto; padding:48px 32px 140px; line-height:1.55; }}
  a {{ color:#58a6ff; }}
  h1,h2,h3 {{ border-bottom:1px solid #21262d; padding-bottom:.3em; }}
  pre {{ background:#161b22; border-radius:6px; padding:14px; overflow:auto; }}
  code {{ background:#161b22; border-radius:6px; padding:2px 5px; }}
  pre code {{ padding:0; }}
  blockquote {{ border-left:3px solid #30363d; color:#8b949e; margin:.4em 0; padding:0 1em; }}
  video {{ max-width:100%; border-radius:8px; }}
  table {{ border-collapse:collapse; }}
  td,th {{ border:1px solid #30363d; padding:6px 12px; }}
  #bar {{ position:fixed; top:0; left:0; right:0; background:#1f6feb; color:#fff;
          font-size:13px; padding:8px 16px; text-align:center; z-index:10; }}
</style>
</head>
<body>
<div id="bar">&#128308; LIVE &middot; {title} &middot; <span id="status">following&hellip;</span></div>
<div id="content">{body}</div>
<script>
  var POLL = {poll};
  var content = document.getElementById('content');
  var last = content.innerHTML;
  var polling = true;
  function setStatus(t, color) {{
    document.getElementById('status').textContent = t;
    if (color) document.getElementById('bar').style.background = color;
  }}
  function isDone(h) {{
    return h.indexOf('Final Summary') !== -1 || h.indexOf('RUN COMPLETE') !== -1;
  }}
  function hasVideo(node) {{
    return node && node.nodeType === 1 && (node.tagName === 'VIDEO' || node.querySelector('video'));
  }}
  // Incremental DOM update. The log is append-only, so keep every block that is
  // unchanged (compared by outerHTML) and only append/replace the tail. A block that
  // contains a <video> is NEVER replaced once rendered -> videos never flicker, reset
  // their playback, or reshuffle, even while the rest of the page keeps updating.
  function morph(newHTML) {{
    var tmp = document.createElement('div');
    tmp.innerHTML = newHTML;
    var newKids = Array.prototype.slice.call(tmp.children);
    var i = 0;
    while (i < content.children.length && i < newKids.length &&
           content.children[i].outerHTML === newKids[i].outerHTML) {{ i++; }}
    for (var j = i; j < newKids.length; j++) {{
      var on = content.children[j];
      if (!on) {{ content.appendChild(newKids[j]); }}
      else if (on.outerHTML !== newKids[j].outerHTML) {{
        if (hasVideo(on)) {{ continue; }}   // never disturb a rendered video block
        content.replaceChild(newKids[j], on);
      }}
    }}
    while (content.children.length > newKids.length &&
           !hasVideo(content.children[content.children.length - 1])) {{
      content.removeChild(content.lastChild);
    }}
  }}
  function tick() {{
    if (!polling) return;
    fetch('content', {{cache:'no-store'}}).then(function(r) {{ return r.text(); }}).then(function(htm) {{
      if (htm !== last) {{
        var nearBottom = (window.innerHeight + window.scrollY) >= (document.body.scrollHeight - 140);
        morph(htm);
        last = htm;
        if (nearBottom) window.scrollTo(0, document.body.scrollHeight);
      }}
      if (isDone(htm)) {{ polling = false; setStatus('\\u2713 run complete \\u2014 refresh stopped', '#238636'); return; }}
      setTimeout(tick, POLL);
    }}).catch(function() {{
      polling = false; setStatus('\\u23f8 disconnected \\u2014 refresh stopped', '#6e7681');
    }});
  }}
  if (isDone(last)) {{ polling = false; setStatus('\\u2713 run complete \\u2014 refresh stopped', '#238636'); }}
  else {{ setTimeout(tick, POLL); }}
</script>
</body>
</html>"""


class LiveHandler(http.server.SimpleHTTPRequestHandler):
    md_path = None  # set on the class before serving

    def _send_html(self, body):
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        route = self.path.split("?")[0]
        if route in ("/", "/index.html"):
            title = os.path.basename(os.path.dirname(self.md_path)) or "run"
            self._send_html(PAGE.format(
                title=html.escape(title), body=render_body(self.md_path), poll=POLL_MS))
        elif route == "/content":
            self._send_html(render_body(self.md_path))
        else:
            # Serve static files (recorded videos, etc.) from the run's output dir
            super().do_GET()

    def log_message(self, *args):
        pass  # keep the terminal quiet


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    output_dir, md_path = resolve_paths(sys.argv[1])
    port = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_PORT

    LiveHandler.md_path = os.path.abspath(md_path)
    handler = functools.partial(LiveHandler, directory=os.path.abspath(output_dir))

    print("Live view of : {}".format(md_path))
    print("Serving root : {}".format(output_dir))
    print("Open         : http://localhost:{}  (smart refresh - stops when the run ends; Ctrl+C to quit)".format(port))
    if _markdown is None:
        print("Tip          : `pip install markdown` for richer rendering (currently raw text).")
    try:
        http.server.ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
