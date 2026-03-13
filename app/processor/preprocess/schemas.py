# from app.core.model import CamelModel
# from datetime import datetime

# class CreatePreprocessingJob(CamelModel):
#     job_id: str
#     current_index: int 
#     processed_count: int
#     skipped_count: int
#     error_count: int
#     target_count: int
    
#     status: str = Field()
#     status_message: str = ""
#     current_file: str
#     continuation_token: str

#     created_at: datetime
#     updated_at: datetime
#     completed_at: datetime