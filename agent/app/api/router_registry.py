# app/api/router_registry.py
import pkgutil
import importlib
from fastapi import APIRouter

def build_master_router() -> APIRouter:
    """
    Dynamically scans all sub-modules located within the routes package, 
    extracts defined APIRouter schemas, and links them onto a unified base route.
    """
    master_router = APIRouter(prefix="/api")
    
    # Target the routes directory dynamically
    import app.api.routes as routes_package
    
    for _, module_name, is_pkg in pkgutil.iter_modules(routes_package.__path__):
        if not is_pkg:
            full_module_path = f"{routes_package.__name__}.{module_name}"
            module = importlib.import_module(full_module_path)
            
            # Verify the module exposes a valid APIRouter deployment handle
            if hasattr(module, "router") and isinstance(module.router, APIRouter):
                master_router.include_router(module.router)
                print(f"==> [Router Registry] Successfully hot-loaded route: '{full_module_path}'")
                
    return master_router