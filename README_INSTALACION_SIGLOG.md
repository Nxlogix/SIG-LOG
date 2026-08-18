# SIG-LOG — instalación rápida

## Windows

```powershell
git clone 
URl:
cd SIG-LOG
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run dashboard\dashboard.py
```

La aplicación debe tener al menos:

- `dashboard/dashboard.py`
- `dashboard/modulos/modulos_operativos_siglog.py`
- `data_warehouse/siglog_dw.db`
- `no_supervisado/resultados/`
- `pca/resultados/`
- `mantenimiento/resultados/`
- `reportes_modelos/`
- `modelos_entrenados/` cuando se utilicen modelos

Si PowerShell bloquea la activación:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Después vuelve a ejecutar `.venv\Scripts\Activate.ps1`.
