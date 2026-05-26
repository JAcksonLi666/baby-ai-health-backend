from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from models import UploadResponse
from config import UPLOAD_DIR, MAX_UPLOAD_SIZE, ALLOWED_EXTENSIONS
from ocr_service import ocr_service
from vector_db import vector_db_service
from datetime import datetime
import uuid
import shutil
import re
from pathlib import Path

router = APIRouter(tags=["档案管理"])


@router.post("/upload/preview")
async def preview_upload(file: UploadFile = File(...)):
    """预识别文件内容，返回识别结果和识别出的日期（不保存）"""
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {file_ext}，支持的类型: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    if file.size and file.size > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"文件过大，最大支持 {MAX_UPLOAD_SIZE / (1024*1024):.1f} MB"
        )

    temp_file_id = f"temp_{uuid.uuid4().hex[:8]}"
    temp_filename = f"{temp_file_id}{file_ext}"
    temp_path = UPLOAD_DIR / temp_filename

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        actual_size = temp_path.stat().st_size
        if actual_size > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=400, detail=f"文件过大，最大支持 {MAX_UPLOAD_SIZE / (1024*1024):.1f} MB")

        if file_ext == '.pdf':
            extracted_text = ocr_service.extract_text_from_pdf(str(temp_path))
        else:
            extracted_text = ocr_service.extract_text_from_image(str(temp_path))

        if not extracted_text:
            return {
                "success": False,
                "message": "未能从文件中提取到文字，请确保图片清晰",
                "extracted_text": "",
                "detected_date": None
            }

        desensitized_text = ocr_service.desensitize_text(extracted_text)
        detected_date = ocr_service.extract_date_from_text(desensitized_text)
        metrics = ocr_service.parse_health_metrics(desensitized_text)

        return {
            "success": True,
            "message": "识别成功",
            "extracted_text": desensitized_text,
            "detected_date": detected_date,
            "metrics": metrics,
            "temp_file_id": temp_file_id
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")
    finally:
        if temp_path.exists():
            temp_path.unlink()


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    record_date: str = Query(None, description="记录日期 (YYYY-MM-DD)"),
    record_type: str = Query("general", description="记录类型: blood_test, urine_test, general")
):
    """上传化验单图片或 PDF，进行 OCR 识别和向量化存储"""
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {file_ext}，支持的类型: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    if file.size and file.size > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"文件过大，最大支持 {MAX_UPLOAD_SIZE / (1024*1024):.1f} MB"
        )

    file_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
    filename = f"{file_id}{file_ext}"
    file_path = UPLOAD_DIR / filename

    if record_date and not re.match(r"^\d{4}-\d{2}-\d{2}$", record_date):
        raise HTTPException(status_code=400, detail="record_date 格式应为 YYYY-MM-DD")

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        actual_size = file_path.stat().st_size
        if actual_size > MAX_UPLOAD_SIZE:
            file_path.unlink()
            raise HTTPException(status_code=400, detail=f"文件过大，最大支持 {MAX_UPLOAD_SIZE / (1024*1024):.1f} MB")

        if file_ext == '.pdf':
            extracted_text = ocr_service.extract_text_from_pdf(str(file_path))
        else:
            extracted_text = ocr_service.extract_text_from_image(str(file_path))
        
        if not extracted_text:
            return UploadResponse(
                success=False,
                file_id=file_id,
                filename=filename,
                message="未能从文件中提取到文字，请确保图片清晰"
            )

        desensitized_text = ocr_service.desensitize_text(extracted_text)
        metrics = ocr_service.parse_health_metrics(desensitized_text)

        final_date = record_date or ocr_service.extract_date_from_text(desensitized_text) or datetime.now().strftime("%Y-%m-%d")

        metadata = {
            "type": record_type,
            "filename": filename,
            "original_filename": file.filename,
            "file_size": file.size or 0,
            "upload_time": datetime.now().isoformat(),
            "record_date": final_date,
            "metrics_count": len(metrics)
        }
        
        storage_text = f"""日期：{metadata['record_date']}
类型：{record_type}
指标：{', '.join([m['name'] for m in metrics]) if metrics else 'N/A'}
内容：{desensitized_text}
"""
        
        vector_db_service.add_record(
            record_id=file_id,
            text=storage_text,
            metadata=metadata,
            date=metadata['record_date']
        )
        
        return UploadResponse(
            success=True,
            file_id=file_id,
            filename=filename,
            message="上传成功，已完成 OCR 识别和向量化存储",
            extracted_text=desensitized_text,
            record_date=final_date
        )
    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")


@router.get("/records")
async def get_records(limit: int = Query(50, ge=1, le=100)):
    """获取所有健康档案记录"""
    try:
        records = vector_db_service.get_all_records(limit=limit)
        stats = vector_db_service.get_collection_stats()
        return {
            "success": True,
            "records": [
                {
                    "id": r["id"],
                    "text": r["text"],
                    "metadata": r["metadata"]
                }
                for r in records
            ],
            "total": stats.get("total_records", 0),
            "showing": len(records)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")


@router.get("/record/{record_id}")
async def get_record(record_id: str):
    """获取指定健康档案详情"""
    try:
        record = vector_db_service.get_record(record_id)
        if record:
            return {"success": True, "record": record}
        else:
            raise HTTPException(status_code=404, detail="记录不存在")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")


@router.delete("/record/{record_id}")
async def delete_record(record_id: str):
    """删除指定健康档案"""
    try:
        success = vector_db_service.delete_record(record_id)
        if success:
            return {"success": True, "message": f"记录 {record_id} 已删除"}
        else:
            raise HTTPException(status_code=500, detail="删除失败")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")


@router.put("/record/{record_id}")
async def update_record(
    record_id: str,
    record_date: str = Query(None),
    record_type: str = Query(None),
    new_text: str = Query(None)
):
    """更新指定健康档案的元数据或内容"""
    try:
        record = vector_db_service.get_record(record_id)
        if not record:
            raise HTTPException(status_code=404, detail="记录不存在")
        
        metadata = record.get("metadata", {})
        if record_date:
            metadata["record_date"] = record_date
        if record_type:
            metadata["type"] = record_type
            metadata["record_type"] = record_type
        
        text = new_text if new_text else record.get("text", "")
        
        if record_date and not new_text:
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if line.startswith("日期："):
                    lines[i] = f"日期：{record_date}"
                    break
            text = '\n'.join(lines)
        
        success = vector_db_service.add_record(
            record_id=record_id,
            text=text,
            metadata=metadata,
            date=metadata.get("record_date")
        )
        
        if success:
            vector_db_service.delete_record(record_id)
            return {
                "success": True,
                "message": f"记录 {record_id} 已更新",
                "record": {"id": record_id, "text": text, "metadata": metadata}
            }
        else:
            raise HTTPException(status_code=500, detail="更新失败")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")


@router.get("/records/filter")
async def filter_records(
    record_type: str = Query(None),
    start_date: str = Query(None),
    end_date: str = Query(None),
    keyword: str = Query(None),
    limit: int = Query(50, ge=1, le=100)
):
    """筛选健康档案记录"""
    try:
        records = vector_db_service.get_all_records(limit=limit)
        
        if record_type:
            records = [r for r in records if r.get("metadata", {}).get("type") == record_type]
        
        if start_date or end_date:
            filtered = []
            for record in records:
                record_date = record.get("metadata", {}).get("record_date", "")
                if record_date:
                    if start_date and record_date < start_date:
                        continue
                    if end_date and record_date > end_date:
                        continue
                filtered.append(record)
            records = filtered
        
        if keyword:
            keyword_lower = keyword.lower()
            records = [
                r for r in records 
                if keyword_lower in r.get("text", "").lower() or 
                   keyword_lower in str(r.get("metadata", {})).lower()
            ]
        
        return {
            "success": True,
            "records": records,
            "total": len(records),
            "filters": {
                "record_type": record_type,
                "start_date": start_date,
                "end_date": end_date,
                "keyword": keyword
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")


@router.get("/records/types")
async def get_record_types():
    """获取所有记录类型统计"""
    try:
        records = vector_db_service.get_all_records(limit=1000)
        
        type_counts = {}
        for record in records:
            record_type = record.get("metadata", {}).get("type", "general")
            type_counts[record_type] = type_counts.get(record_type, 0) + 1
        
        type_names = {
            "blood_test": "血液检测",
            "urine_test": "尿液检测",
            "general": "常规记录"
        }
        
        result = []
        for record_type, count in type_counts.items():
            result.append({
                "type": record_type,
                "name": type_names.get(record_type, record_type),
                "count": count
            })
        
        return {
            "success": True,
            "types": result,
            "total_records": len(records)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")
