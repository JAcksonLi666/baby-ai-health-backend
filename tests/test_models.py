"""
Unit tests for Pydantic models in models.py.

Tests validation of all model classes including:
- Record models (Sleep, Diaper, Cry, Feeding, Growth)
- ReminderRecord
- LabReportParseRequest
- SymptomCheckRequest
- ChatMessageCreate
- KnowledgeEntryCreate
"""
import pytest
from datetime import datetime
from pydantic import ValidationError

# Import all models
from models import (
    # Enums
    SleepType, DiaperType, PoopColor, PoopConsistency, Amount, CryReason,
    FeedingType, BreastSide, GrowthRecordCreate, GrowthRecordUpdate,
    ReminderType, ReminderStatus, RepeatType, ReportType, MessageRole,
    # Record Create models
    SleepRecordCreate, DiaperRecordCreate, CryRecordCreate,
    FeedingRecordCreate, GrowthRecordCreate, ReminderRecordCreate,
    # Request models
    LabReportParseRequest, SymptomCheckRequest, ChatMessageCreate,
    KnowledgeEntryCreate,
)


class TestSleepRecord:
    """Test cases for SleepRecord models."""

    def test_sleep_record_create_valid(self):
        """Test valid sleep record creation."""
        record = SleepRecordCreate(
            start_time="2024-01-15 20:00",
            sleep_type=SleepType.night,
            notes="Slept well",
            is_ongoing=False,
        )
        assert record.start_time == "2024-01-15 20:00"
        assert record.sleep_type == SleepType.night
        assert record.notes == "Slept well"
        assert record.is_ongoing is False

    def test_sleep_record_create_with_nap(self):
        """Test sleep record with nap type."""
        record = SleepRecordCreate(
            start_time="2024-01-15 14:00",
            sleep_type=SleepType.nap,
        )
        assert record.sleep_type == SleepType.nap

    def test_sleep_record_create_required_start_time(self):
        """Test that start_time is required."""
        with pytest.raises(ValidationError) as exc_info:
            SleepRecordCreate()
        errors = exc_info.value.errors()
        assert any("start_time" in str(e) for e in errors)


class TestDiaperRecord:
    """Test cases for DiaperRecord models."""

    def test_diaper_record_create_valid(self):
        """Test valid diaper record creation."""
        record = DiaperRecordCreate(
            time="2024-01-15 10:30",
            diaper_type=DiaperType.poop,
            poop_color=PoopColor.yellow,
            poop_consistency=PoopConsistency.soft,
            amount=Amount.medium,
        )
        assert record.time == "2024-01-15 10:30"
        assert record.diaper_type == DiaperType.poop
        assert record.poop_color == PoopColor.yellow

    def test_diaper_record_both_type(self):
        """Test diaper record with both type."""
        record = DiaperRecordCreate(
            time="2024-01-15 10:30",
            diaper_type=DiaperType.both,
        )
        assert record.diaper_type == DiaperType.both

    def test_diaper_record_create_required_fields(self):
        """Test required fields for diaper record."""
        with pytest.raises(ValidationError) as exc_info:
            DiaperRecordCreate()
        errors = exc_info.value.errors()
        assert any("time" in str(e) or "diaper_type" in str(e) for e in errors)

    def test_diaper_record_all_poop_colors(self):
        """Test all valid poop colors."""
        valid_colors = [PoopColor.yellow, PoopColor.green, PoopColor.brown,
                       PoopColor.black, PoopColor.red, PoopColor.white, PoopColor.orange]
        for color in valid_colors:
            record = DiaperRecordCreate(
                time="2024-01-15 10:30",
                diaper_type=DiaperType.poop,
                poop_color=color,
            )
            assert record.poop_color == color


class TestCryRecord:
    """Test cases for CryRecord models."""

    def test_cry_record_create_valid(self):
        """Test valid cry record creation."""
        record = CryRecordCreate(
            start_time="2024-01-15 08:00",
            end_time="2024-01-15 08:30",
            reason=CryReason.hungry,
            intensity=3,
            soothing_method="Feed",
        )
        assert record.start_time == "2024-01-15 08:00"
        assert record.reason == CryReason.hungry
        assert record.intensity == 3

    def test_cry_record_intensity_range(self):
        """Test cry record intensity validation (1-5)."""
        for i in range(1, 6):
            record = CryRecordCreate(
                start_time="2024-01-15 08:00",
                intensity=i,
            )
            assert record.intensity == i

        with pytest.raises(ValidationError):
            CryRecordCreate(start_time="2024-01-15 08:00", intensity=0)

        with pytest.raises(ValidationError):
            CryRecordCreate(start_time="2024-01-15 08:00", intensity=6)

    def test_cry_record_all_reasons(self):
        """Test all valid cry reasons."""
        valid_reasons = [CryReason.hungry, CryReason.sleepy, CryReason.diaper,
                        CryReason.discomfort, CryReason.pain, CryReason.lonely,
                        CryReason.overstimulated, CryReason.unknown]
        for reason in valid_reasons:
            record = CryRecordCreate(
                start_time="2024-01-15 08:00",
                reason=reason,
            )
            assert record.reason == reason


class TestFeedingRecord:
    """Test cases for FeedingRecord models."""

    def test_feeding_record_breast(self):
        """Test breast feeding record."""
        record = FeedingRecordCreate(
            time="2024-01-15 09:00",
            feeding_type=FeedingType.breast,
            duration_minutes=15,
            breast_side=BreastSide.left,
        )
        assert record.feeding_type == FeedingType.breast
        assert record.duration_minutes == 15
        assert record.breast_side == BreastSide.left

    def test_feeding_record_formula(self):
        """Test formula feeding record."""
        record = FeedingRecordCreate(
            time="2024-01-15 09:00",
            feeding_type=FeedingType.formula,
            amount_ml=120.0,
        )
        assert record.feeding_type == FeedingType.formula
        assert record.amount_ml == 120.0

    def test_feeding_record_solid_food(self):
        """Test solid food feeding record."""
        record = FeedingRecordCreate(
            time="2024-01-15 12:00",
            feeding_type=FeedingType.solid,
            solid_food="南瓜泥",
        )
        assert record.feeding_type == FeedingType.solid
        assert record.solid_food == "南瓜泥"

    def test_feeding_record_amount_validation(self):
        """Test amount validation (must be >= 0)."""
        record = FeedingRecordCreate(
            time="2024-01-15 09:00",
            feeding_type=FeedingType.formula,
            amount_ml=0,
        )
        assert record.amount_ml == 0

        with pytest.raises(ValidationError):
            FeedingRecordCreate(
                time="2024-01-15 09:00",
                feeding_type=FeedingType.formula,
                amount_ml=-10,
            )

    def test_feeding_record_duration_validation(self):
        """Test duration validation (must be >= 0)."""
        record = FeedingRecordCreate(
            time="2024-01-15 09:00",
            feeding_type=FeedingType.breast,
            duration_minutes=0,
        )
        assert record.duration_minutes == 0


class TestGrowthRecord:
    """Test cases for GrowthRecord models."""

    def test_growth_record_valid(self):
        """Test valid growth record creation."""
        record = GrowthRecordCreate(
            record_date="2024-01-15",
            weight_kg=7.5,
            height_cm=68.0,
            head_circumference_cm=42.0,
            temperature_c=36.8,
        )
        assert record.record_date == "2024-01-15"
        assert record.weight_kg == 7.5
        assert record.height_cm == 68.0

    def test_growth_record_weight_range(self):
        """Test weight validation range (0-150 kg)."""
        record = GrowthRecordCreate(record_date="2024-01-15", weight_kg=0)
        assert record.weight_kg == 0

        record = GrowthRecordCreate(record_date="2024-01-15", weight_kg=150)
        assert record.weight_kg == 150

        with pytest.raises(ValidationError):
            GrowthRecordCreate(record_date="2024-01-15", weight_kg=-1)

        with pytest.raises(ValidationError):
            GrowthRecordCreate(record_date="2024-01-15", weight_kg=200)

    def test_growth_record_height_range(self):
        """Test height validation range (0-250 cm)."""
        record = GrowthRecordCreate(record_date="2024-01-15", height_cm=0)
        assert record.height_cm == 0

        record = GrowthRecordCreate(record_date="2024-01-15", height_cm=250)
        assert record.height_cm == 250

        with pytest.raises(ValidationError):
            GrowthRecordCreate(record_date="2024-01-15", height_cm=-1)

        with pytest.raises(ValidationError):
            GrowthRecordCreate(record_date="2024-01-15", height_cm=300)

    def test_growth_record_temperature_range(self):
        """Test temperature validation range (35-42 C)."""
        record = GrowthRecordCreate(record_date="2024-01-15", temperature_c=35.0)
        assert record.temperature_c == 35.0

        record = GrowthRecordCreate(record_date="2024-01-15", temperature_c=42.0)
        assert record.temperature_c == 42.0

        with pytest.raises(ValidationError):
            GrowthRecordCreate(record_date="2024-01-15", temperature_c=34.9)

        with pytest.raises(ValidationError):
            GrowthRecordCreate(record_date="2024-01-15", temperature_c=42.1)


class TestReminderRecord:
    """Test cases for ReminderRecord models."""

    def test_reminder_record_valid(self):
        """Test valid reminder record creation."""
        record = ReminderRecordCreate(
            title="疫苗接种",
            reminder_type=ReminderType.vaccine,
            reminder_date="2024-02-15",
            reminder_time="09:00",
        )
        assert record.title == "疫苗接种"
        assert record.reminder_type == ReminderType.vaccine
        assert record.status == ReminderStatus.pending

    def test_reminder_record_title_length(self):
        """Test title length validation (1-200 characters)."""
        record = ReminderRecordCreate(
            title="A",
            reminder_type=ReminderType.checkup,
            reminder_date="2024-02-15",
        )
        assert len(record.title) == 1

        with pytest.raises(ValidationError):
            ReminderRecordCreate(
                title="A" * 201,
                reminder_type=ReminderType.checkup,
                reminder_date="2024-02-15",
            )

    def test_reminder_record_all_types(self):
        """Test all reminder types."""
        types = [ReminderType.vaccine, ReminderType.checkup, ReminderType.feeding,
                 ReminderType.medicine, ReminderType.other]
        for rtype in types:
            record = ReminderRecordCreate(
                title="Test",
                reminder_type=rtype,
                reminder_date="2024-02-15",
            )
            assert record.reminder_type == rtype

    def test_reminder_record_all_statuses(self):
        """Test all reminder statuses."""
        statuses = [ReminderStatus.pending, ReminderStatus.completed,
                   ReminderStatus.overdue, ReminderStatus.cancelled]
        for status in statuses:
            record = ReminderRecordCreate(
                title="Test",
                reminder_type=ReminderType.other,
                reminder_date="2024-02-15",
                status=status,
            )
            assert record.status == status

    def test_reminder_record_all_repeat_types(self):
        """Test all repeat types."""
        repeat_types = [RepeatType.none, RepeatType.daily, RepeatType.weekly, RepeatType.monthly]
        for rtype in repeat_types:
            record = ReminderRecordCreate(
                title="Test",
                reminder_type=ReminderType.other,
                reminder_date="2024-02-15",
                repeat_type=rtype,
            )
            assert record.repeat_type == rtype


class TestLabReportParseRequest:
    """Test cases for LabReportParseRequest model."""

    def test_lab_report_parse_request_valid(self):
        """Test valid lab report parse request."""
        request = LabReportParseRequest(
            text="WBC 5.2 x10^9/L 4-10",
            report_type=ReportType.blood,
            age_months=12,
        )
        assert request.text == "WBC 5.2 x10^9/L 4-10"
        assert request.report_type == ReportType.blood
        assert request.age_months == 12

    def test_lab_report_parse_request_required_text(self):
        """Test that text field is required."""
        with pytest.raises(ValidationError) as exc_info:
            LabReportParseRequest()
        errors = exc_info.value.errors()
        assert any("text" in str(e) for e in errors)

    def test_lab_report_parse_request_age_range(self):
        """Test age_months validation range (0-144 months)."""
        request = LabReportParseRequest(text="test", age_months=0)
        assert request.age_months == 0

        request = LabReportParseRequest(text="test", age_months=144)
        assert request.age_months == 144

        with pytest.raises(ValidationError):
            LabReportParseRequest(text="test", age_months=-1)

        with pytest.raises(ValidationError):
            LabReportParseRequest(text="test", age_months=145)

    def test_lab_report_parse_request_all_report_types(self):
        """Test all valid report types."""
        types = [ReportType.auto, ReportType.blood, ReportType.urine,
                 ReportType.liver, ReportType.kidney, ReportType.blood_routine,
                 ReportType.urine_routine, ReportType.liver_function, ReportType.kidney_function]
        for rtype in types:
            request = LabReportParseRequest(text="test", report_type=rtype)
            assert request.report_type == rtype

    def test_lab_report_parse_request_default_values(self):
        """Test default values."""
        request = LabReportParseRequest(text="test")
        assert request.report_type == ReportType.auto
        assert request.age_months == 6


class TestSymptomCheckRequest:
    """Test cases for SymptomCheckRequest model."""

    def test_symptom_check_request_valid(self):
        """Test valid symptom check request."""
        request = SymptomCheckRequest(
            symptoms=["发烧", "咳嗽"],
            age_months=12,
            duration_days=3,
            severity=3,
        )
        assert request.symptoms == ["发烧", "咳嗽"]
        assert request.age_months == 12
        assert request.duration_days == 3
        assert request.severity == 3

    def test_symptom_check_request_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError) as exc_info:
            SymptomCheckRequest()
        errors = exc_info.value.errors()
        assert any("symptoms" in str(e) or "age_months" in str(e) for e in errors)

    def test_symptom_check_request_age_range(self):
        """Test age_months validation range (0-144 months)."""
        request = SymptomCheckRequest(symptoms=["发烧"], age_months=0)
        assert request.age_months == 0

        request = SymptomCheckRequest(symptoms=["发烧"], age_months=144)
        assert request.age_months == 144

        with pytest.raises(ValidationError):
            SymptomCheckRequest(symptoms=["发烧"], age_months=-1)

        with pytest.raises(ValidationError):
            SymptomCheckRequest(symptoms=["发烧"], age_months=150)

    def test_symptom_check_request_duration_days(self):
        """Test duration_days validation (>= 0)."""
        request = SymptomCheckRequest(symptoms=["发烧"], age_months=12, duration_days=0)
        assert request.duration_days == 0

        with pytest.raises(ValidationError):
            SymptomCheckRequest(symptoms=["发烧"], age_months=12, duration_days=-1)

    def test_symptom_check_request_severity_range(self):
        """Test severity validation range (1-5)."""
        for i in range(1, 6):
            request = SymptomCheckRequest(symptoms=["发烧"], age_months=12, severity=i)
            assert request.severity == i

        with pytest.raises(ValidationError):
            SymptomCheckRequest(symptoms=["发烧"], age_months=12, severity=0)

        with pytest.raises(ValidationError):
            SymptomCheckRequest(symptoms=["发烧"], age_months=12, severity=6)


class TestChatMessageCreate:
    """Test cases for ChatMessageCreate model."""

    def test_chat_message_create_valid(self):
        """Test valid chat message creation."""
        message = ChatMessageCreate(
            session_id="chat_123",
            role=MessageRole.user,
            content="Hello",
        )
        assert message.session_id == "chat_123"
        assert message.role == MessageRole.user
        assert message.content == "Hello"

    def test_chat_message_create_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError) as exc_info:
            ChatMessageCreate()
        errors = exc_info.value.errors()
        assert any("session_id" in str(e) or "role" in str(e) or "content" in str(e) for e in errors)

    def test_chat_message_create_all_roles(self):
        """Test all valid message roles."""
        roles = [MessageRole.user, MessageRole.assistant, MessageRole.system]
        for role in roles:
            message = ChatMessageCreate(
                session_id="chat_123",
                role=role,
                content="Test",
            )
            assert message.role == role

    def test_chat_message_create_invalid_role(self):
        """Test invalid role value."""
        with pytest.raises(ValidationError):
            ChatMessageCreate(
                session_id="chat_123",
                role="invalid_role",
                content="Test",
            )


class TestKnowledgeEntryCreate:
    """Test cases for KnowledgeEntryCreate model."""

    def test_knowledge_entry_create_valid(self):
        """Test valid knowledge entry creation."""
        entry = KnowledgeEntryCreate(
            title="Test Title",
            content="Test content",
            source="Test source",
            keywords=["test", "keyword"],
            category="test",
        )
        assert entry.title == "Test Title"
        assert entry.content == "Test content"
        assert entry.keywords == ["test", "keyword"]

    def test_knowledge_entry_create_minimal(self):
        """Test minimal knowledge entry (only required fields)."""
        entry = KnowledgeEntryCreate(
            title="Test Title",
            content="Test content",
        )
        assert entry.title == "Test Title"
        assert entry.content == "Test content"
        assert entry.source is None
        assert entry.keywords == []

    def test_knowledge_entry_create_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError) as exc_info:
            KnowledgeEntryCreate()
        errors = exc_info.value.errors()
        assert any("title" in str(e) or "content" in str(e) for e in errors)
