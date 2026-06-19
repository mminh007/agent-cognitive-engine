# app/mcp/domains/core_tools.py
def calculate_execution_time_logic(milliseconds: int) -> str:
    """Pure logic for converting execution milliseconds."""
    minutes = milliseconds // 60000
    seconds = (milliseconds % 60000) // 1000
    return f"{minutes} minutes {seconds} seconds"