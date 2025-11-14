"""
ENHANCED: FastAPI Backend with Helper Endpoints
================================================
Added GET endpoints for wells, zones, and fields to help users
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import sqlite3
import anthropic
import os
from pathlib import Path
from dotenv import load_dotenv

# =============================================================================
# Models
# =============================================================================

class ChatRequest(BaseModel):
    question: str
    company: str


class ChatResponse(BaseModel):
    sql: Optional[str] = None
    columns: Optional[List[str]] = None
    rows: Optional[List] = None
    error: Optional[str] = None


class HelperData(BaseModel):
    """Response model for helper dropdown data"""
    wells: List[str]
    zones: List[str]
    fields: List[str]


# =============================================================================
# Database Configuration
# =============================================================================

class DatabaseConfig:
    def __init__(self, company_name: str, db_path: str, table_mapping: dict, 
                 column_mapping: dict, well_types: dict):
        self.company_name = company_name
        self.db_path = Path(db_path)
        self.table_mapping = table_mapping
        self.column_mapping = column_mapping
        self.well_types = well_types
    
    def get_real_table_name(self, standard_table_name: str) -> str:
        return self.table_mapping.get(standard_table_name, standard_table_name)
    
    def get_real_column_name(self, standard_column_name: str) -> str:
        return self.column_mapping.get(standard_column_name, standard_column_name)
    
    def get_helper_data(self) -> dict:
        """Get distinct wells, zones, and fields for autocomplete"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            fact_table = self.get_real_table_name("fact_table")
            well_name_col = self.get_real_column_name("well_name")
            zone_col = self.get_real_column_name("zone")
            field_col = self.get_real_column_name("field")
            
            # Get distinct wells
            cursor.execute(f"SELECT DISTINCT {well_name_col} FROM {fact_table} WHERE {well_name_col} IS NOT NULL ORDER BY {well_name_col}")
            wells = [row[0] for row in cursor.fetchall()]
            
            # Get distinct zones
            cursor.execute(f"SELECT DISTINCT {zone_col} FROM {fact_table} WHERE {zone_col} IS NOT NULL ORDER BY {zone_col}")
            zones = [row[0] for row in cursor.fetchall()]
            
            # Get distinct fields
            cursor.execute(f"SELECT DISTINCT {field_col} FROM {fact_table} WHERE {field_col} IS NOT NULL ORDER BY {field_col}")
            fields = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            
            return {
                "wells": wells,
                "zones": zones,
                "fields": fields
            }
        except Exception as e:
            print(f"❌ Error getting helper data: {e}")
            return {"wells": [], "zones": [], "fields": []}


class SQLGenerator:
    def __init__(self, config: DatabaseConfig):
        self.config = config
        
        api_key = os.getenv("ANTHROPIC_API_KEY")

        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables.")
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def generate_enhanced_prompt(self) -> str:
        fact_table = self.config.get_real_table_name("fact_table")
        production_table = self.config.get_real_table_name("production_table")
        injection_table = self.config.get_real_table_name("injection_table")
        fluid_table = self.config.get_real_table_name("fluid_table")
        
        well_id = self.config.get_real_column_name("well_id")
        well_name = self.config.get_real_column_name("well_name")
        zone = self.config.get_real_column_name("zone")
        field = self.config.get_real_column_name("field")
        well_type = self.config.get_real_column_name("well_type")
        date_col = self.config.get_real_column_name("date")
        oil_prod = self.config.get_real_column_name("oil_production")
        water_prod = self.config.get_real_column_name("water_production")
        gas_prod = self.config.get_real_column_name("gas_production")
        injection_rate = self.config.get_real_column_name("injection_rate")
        fluid_level = self.config.get_real_column_name("fluid_level")
        run_time = self.config.get_real_column_name("run_time")
        pip = self.config.get_real_column_name("pip")
        
        producer_value = self.config.well_types["producer"]
        injector_value = self.config.well_types["injector"]
        
        prompt = f"""
You are an expert SQL assistant for the {self.config.company_name} oil & gas database.

### Database Schema:

1. **{fact_table}**: Main reference table with well information.
   - {well_id}: Primary key (used for JOINs)
   - {well_name}: Well name/bore identifier
   - {zone}: Reservoir zone
   - {field}: Field name
   - {well_type}: Well type ('{producer_value}' = producer, '{injector_value}' = injector)

2. **{production_table}**: Daily production records.
   - {well_id}: Foreign key, links to {fact_table} (use for JOINs)
   - {date_col}: Production date (YYYY-MM-DD)
   - {oil_prod}: Oil production volume
   - {water_prod}: Water production volume
   - {gas_prod}: Gas production volume
   - {run_time}: Operating hours

3. **{injection_table}**: Daily injection records.
   - {well_id}: Foreign key, links to {fact_table} (use for JOINs)
   - {date_col}: Injection date (YYYY-MM-DD)
   - {injection_rate}: Injection rate

4. **{fluid_table}**: Daily fluid level measurements.
   - {well_id}: Foreign key, links to {fact_table} (use for JOINs)
   - {date_col}: Measurement date
   - {fluid_level}: Fluid level measurement
   -{pip}: presse in psi bottom hole pressure (pwf)

### Special Instructions:

1. **For table structure**: SELECT name FROM sqlite_master WHERE type='table';
2. **For column info**: PRAGMA table_info(table_name);
3. **String values**: Always use single quotes, e.g., WHERE {well_type} = '{producer_value}'
4. **Date filtering**: Use date('now'), date('now', '-1 day'), date('now', '-7 days')
5. **JOIN operations**: ALWAYS join production/injection/fluid tables with {fact_table} using {well_id}
6. **Aggregations**: Use SUM(), AVG(), COUNT(DISTINCT) as appropriate with GROUP BY
7. **Table aliases**: Use dp ({production_table}), di ({injection_table}), df ({fluid_table}), h ({fact_table})
8. **Counting unique wells**: JOIN with {fact_table} and count {well_name}, not {well_id}
9. **For totals**: Use SUM() directly, don't multiply by {run_time} unless explicitly asked

### Examples:

1. Count unique wells:
   SELECT COUNT(DISTINCT {well_id}) FROM {fact_table};

2. Production for January 2024 with details:
   SELECT h.{well_name}, h.{zone}, dp.{date_col}, dp.{oil_prod}
   FROM {production_table} dp
   JOIN {fact_table} h ON dp.{well_id} = h.{well_id}
   WHERE dp.{date_col} BETWEEN '2024-01-01' AND '2024-01-31';

3. Total production yesterday for field 'Se Gendi':
   SELECT SUM(dp.{oil_prod}) AS total
   FROM {production_table} dp
   JOIN {fact_table} h ON dp.{well_id} = h.{well_id}
   WHERE dp.{date_col} = date('now', '-1 day') AND h.{field} = 'Se Gendi';

4. Count wells with production > 0 today:
   SELECT COUNT(DISTINCT h.{well_name})
   FROM {production_table} dp
   JOIN {fact_table} h ON dp.{well_id} = h.{well_id}
   WHERE dp.{date_col} = date('now') AND dp.{oil_prod} > 0;

5. Total injection by field:
   SELECT h.{field}, SUM(di.{injection_rate}) AS total
   FROM {injection_table} di
   JOIN {fact_table} h ON di.{well_id} = h.{well_id}
   GROUP BY h.{field};

6. Total lifetime production:
   SELECT SUM({oil_prod}) FROM {production_table};

7. List producer wells:
   SELECT {well_name} FROM {fact_table} WHERE {well_type} = '{producer_value}';

8. List injector wells:
   SELECT {well_name} FROM {fact_table} WHERE {well_type} = '{injector_value}';

9. Top 5 wells by production this month:
   SELECT h.{well_name}, SUM(dp.{oil_prod}) AS total
   FROM {production_table} dp
   JOIN {fact_table} h ON dp.{well_id} = h.{well_id}
   WHERE dp.{date_col} >= date('now', 'start of month')
   GROUP BY h.{well_name}
   ORDER BY total DESC LIMIT 5;

10. Average production by zone:
    SELECT h.{zone}, AVG(dp.{oil_prod}) AS avg_oil
    FROM {production_table} dp
    JOIN {fact_table} h ON dp.{well_id} = h.{well_id}
    GROUP BY h.{zone};

Generate ONLY the SQL query, no explanations.
"""
        return prompt
    
    def get_claude_response(self, question: str) -> Optional[str]:
        try:
            enhanced_prompt = self.generate_enhanced_prompt()
            
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                temperature=0.2,
                messages=[
                    {"role": "user", "content": f"{enhanced_prompt}\n\nUser Question: {question}\n\nSQL Query:"}
                ]
            )
            
            sql = message.content[0].text.strip()
            sql = sql.replace("```sql", "").replace("```", "").strip()
            print(f"🧠 Generated SQL: {sql}")
            return sql
        except Exception as e:
            print(f"❌ Claude Error: {e}")
            return None


class DatabaseExecutor:
    def __init__(self, config: DatabaseConfig):
        self.config = config
    
    def execute_sql(self, sql: str):
        try:
            conn = sqlite3.connect(self.config.db_path)
            cur = conn.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            cols = [desc[0] for desc in cur.description] if cur.description else []
            conn.close()
            return cols, rows, None
        except sqlite3.Error as e:
            return None, None, str(e)
    
    def validate_sql(self, query: str) -> bool:
        forbidden = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", "TRUNCATE"]
        return not any(word in query.upper() for word in forbidden)


def create_petrosila_config() -> DatabaseConfig:
    return DatabaseConfig(
        company_name="petrosila",
        db_path=Path(__file__).parent / "petrosila.db",
        table_mapping={
            "fact_table": "header_id",
            "production_table": "daily_production",
            "injection_table": "daily_injection",
            "fluid_table": "fluid_level"
        },
        column_mapping={
            "well_id": "well_zone",
            "well_name": "well_bore",
            "zone": "zone",
            "field": "field",
            "well_type": "type",
            "date": "date",
            "oil_production": "net_oil",
            "water_production": "water",
            "gas_production": "gas",
            "run_time": "run_time",
            "injection_rate": "inj_rate",
            "fluid_level": "nlap",
            "pip":"pip"
        },
        well_types={
            "producer": "producer",
            "injector": "WI"
        }
    )


def create_alamein_config() -> DatabaseConfig:
    return DatabaseConfig(
        company_name="alamein",
        db_path=Path(__file__).parent / "data" / "alamein_db.sqlite3",
        table_mapping={
            "fact_table": "header_id",
            "production_table": "daily_production",
            "injection_table": "daily_injection",
            "fluid_table": "view_dfl"
        },
        column_mapping={
            "well_id": "unique_id",
            "well_name": "well_bore",
            "zone": "zone",
            "field": "field",
            "well_type": "type",
            "date": "date",
            "oil_production": "net",
            "gross": "gross",
            "wc":'wc',
            
            "run_time": "hrs",
            "injection_rate": "inj_rate",
            "fluid_level": "dfl",
            "pip": "pi"
            

        },
        well_types={
            "producer": "producer",
            "injector": "injector"
        }
    )


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="SQL Chatbot API - Enhanced & Fixed",
    description="Natural language to SQL with comprehensive examples",
    version="2.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "status": "running",
        "message": "SQL Chatbot API with Helper Endpoints!",
        "version": "2.2.0",
        "endpoints": {
            "/chat": "POST - Send query",
            "/helpers/{company}": "GET - Get wells, zones, fields"
        }
    }


@app.get("/companies")
def get_companies():
    return {"companies": ["petrosila", "alamein"]}


@app.get("/helpers/{company}", response_model=HelperData)
def get_helpers(company: str):
    """
    Get distinct wells, zones, and fields for a company
    This helps users with autocomplete/suggestions
    """
    try:
        if company.lower() == "petrosila":
            config = create_petrosila_config()
        elif company.lower() == "alamein":
            config = create_alamein_config()
        else:
            raise HTTPException(status_code=400, detail=f"Unknown company: {company}")
        
        helper_data = config.get_helper_data()
        
        return HelperData(
            wells=helper_data["wells"],
            zones=helper_data["zones"],
            fields=helper_data["fields"]
        )
        
    except Exception as e:
        import traceback
        error_detail = f"Server error: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        if request.company.lower() == "petrosila":
            config = create_petrosila_config()
        elif request.company.lower() == "alamein":
            config = create_alamein_config()
        else:
            raise HTTPException(status_code=400, detail=f"Unknown company: {request.company}")
        
        sql_generator = SQLGenerator(config)
        db_executor = DatabaseExecutor(config)
        
        sql = sql_generator.get_claude_response(request.question)
        
        if not sql:
            return ChatResponse(error="Failed to generate SQL query")
        
        if not db_executor.validate_sql(sql):
            return ChatResponse(sql=sql, error="Generated SQL contains forbidden operations")
        
        columns, rows, error = db_executor.execute_sql(sql)
        
        if error:
            return ChatResponse(sql=sql, error=f"SQL execution error: {error}")
        
        return ChatResponse(sql=sql, columns=columns, rows=rows, error=None)
        
    except Exception as e:
        import traceback
        error_detail = f"Server error: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        return ChatResponse(error=f"Server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)