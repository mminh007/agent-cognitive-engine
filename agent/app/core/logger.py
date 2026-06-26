# app/core/logger.py
import logging
import os
from logging.handlers import TimedRotatingFileHandler
from app.core.settings import settings

# --- Custom Log Level Registration ---
PROCESS_LEVEL = 25
logging.addLevelName(PROCESS_LEVEL, "PROCESS")

def _logger_process(self, message, *args, **kws):
    if self.isEnabledFor(PROCESS_LEVEL):
        self._log(PROCESS_LEVEL, message, args, **kws)

logging.Logger.process = _logger_process
# ---------------------------------------

class HtmlFormatter(logging.Formatter):
    CSS_STYLE = """
<style>
 body { 
    font-family: monospace; 
    background-color: #ffffff; 
    color: #2c3e50;            
    margin: 0; 
    padding: 10px; 
}
.log-line { 
    margin-bottom: 15px; 
    padding: 8px; 
    border-bottom: 1px solid #e0e0e0; 
    white-space: pre-wrap; 
    word-wrap: break-word; 
}
.timestamp { 
    color: #7f8c8d;
    margin-right: 10px; 
}
.level { 
    font-weight: bold; 
    margin-right: 10px; 
}
.success { 
    color: #1e7e34; 
    font-weight: bold; 
    background-color: #d4edda; 
    padding: 2px 5px; 
    border-radius: 3px; 
}
.processing { 
    color: #0056b3; 
}
.error { 
    color: #bd2130; 
    font-weight: bold; 
    background-color: #f8d7da; 
    padding: 2px 5px; 
    border-radius: 3px; 
}
.warning { 
    color: #856404; 
    font-weight: bold; 
    background-color: #fff3cd; 
    padding: 2px 5px; 
    border-radius: 3px; 
}
.summary-box { 
    margin-top: 30px; 
    border-top: 3px solid #6c757d; 
    background: #f8f9fa; 
    padding: 20px; 
    border-radius: 0 0 8px 8px; 
    color: #212529; 
    white-space: pre-wrap; 
}
.summary-title { 
    font-size: 20px; 
    font-weight: bold; 
    margin-bottom: 15px; 
    color: #212529; 
    text-transform: uppercase; 
}
.default { 
    color: #2c3e50; 
}

</style>
"""

    def format(self, record):
        msg = super().format(record)
        msg_str = str(msg)
        
        css_class = "default"
        
        if "summary" in msg_str.lower() or "tổng hợp" in msg_str.lower():
            return f'<div class="summary-box"><div class="summary-title">{record.levelname} SUMMARY</div>{msg_str}</div>'
            
        if "[PROCESS]" in msg_str or record.levelname == "PROCESS":
            css_class = "processing"
        elif "[ERROR]" in msg_str or record.levelname == "ERROR":
            css_class = "error"
        elif "[WARNING]" in msg_str or record.levelname == "WARNING":
            css_class = "warning"
        elif "[INFO]" in msg_str or record.levelname == "INFO":
            css_class = "success"
            
        return f'<div class="log-line {css_class}"><span class="timestamp">{self.formatTime(record, self.datefmt)}</span> <span class="level">[{record.levelname}]</span> {msg_str}</div>'


class DailyAndSizeRotatingFileHandler(TimedRotatingFileHandler):
    """
    Rotates on daily boundaries and when the file exceeds a certain size.
    """
    def __init__(self, filename, when='midnight', interval=1, backupCount=0, encoding=None, delay=False, utc=False, atTime=None, maxBytes=0):
        super().__init__(filename, when, interval, backupCount, encoding, delay, utc, atTime)
        self.maxBytes = maxBytes
        self._inject_css_header()

    def _inject_css_header(self):
        if self.stream:
            self.stream.seek(0, 2)
            if self.stream.tell() == 0:
                self.stream.write(HtmlFormatter.CSS_STYLE + "\n")
                self.stream.flush()

    def shouldRollover(self, record):
        if super().shouldRollover(record):
            return 1
        if self.maxBytes > 0:
            msg = "%s\n" % self.format(record)
            self.stream.seek(0, 2)
            if self.stream.tell() + len(msg.encode(self.encoding or 'utf-8')) >= self.maxBytes:
                return 1
        return 0

    def doRollover(self):
        super().doRollover()
        self._inject_css_header()


def setup_app_logger(name: str) -> logging.Logger:
    """
    Initializes a highly resilient structural logger utilizing a dual-sink output framework:
    1. StreamHandler (Console Output for instantaneous Docker Telemetry tracking)
    2. DailyAndSizeRotatingFileHandler (HTML Disk logging bounded by dynamic byte-size & daily rotation)
    """
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    
    log_directory = settings.logs.dir
    os.makedirs(log_directory, exist_ok=True)
    log_filepath = os.path.join(log_directory, "agent_platform.html")
    
    text_format = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] [Thread:%(thread)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    html_format = HtmlFormatter(
        fmt="[%(name)s] [Thread:%(thread)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(text_format)
    logger.addHandler(console_handler)
    
    file_handler = DailyAndSizeRotatingFileHandler(
        filename=log_filepath,
        when="midnight",
        interval=1,
        backupCount=settings.logs.backup_count,
        encoding="utf-8",
        maxBytes=settings.logs.max_bytes
    )
    file_handler.setFormatter(html_format)
    logger.addHandler(file_handler)
    
    return logger