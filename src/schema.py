from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


WorkingType = Literal["remote", "hybrid", "onsite"]
SowType = Literal["materialbased", "timebased"]
Language = Literal["en", "nl"]


class Boundaries(BaseModel):
    includedActivities: List[str] = Field(min_length=1)
    outOfScope: List[str] = Field(min_length=1)

    @field_validator("includedActivities", "outOfScope")
    @classmethod
    def validate_string_list(cls, value: List[str]) -> List[str]:
        cleaned = []
        for item in value:
            item = str(item).strip()
            if item:
                cleaned.append(item[:255])
        if not cleaned:
            raise ValueError("List must contain at least one non-empty item.")
        return cleaned


class Budget(BaseModel):
    costestimate: Optional[int] = None
    hourlyrate: Optional[int] = None
    averageweeklyhours: Optional[int] = None

    @field_validator("costestimate", "hourlyrate", "averageweeklyhours")
    @classmethod
    def non_negative_ints(cls, value: Optional[int]) -> Optional[int]:
        if value is None:
            return value
        if not isinstance(value, int):
            raise ValueError("Budget values must be integers.")
        if value < 0:
            raise ValueError("Budget values must be non-negative.")
        return value


class Location(BaseModel):
    workingtype: WorkingType
    worklocation: Optional[str] = None

    @field_validator("worklocation")
    @classmethod
    def trim_worklocation(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value[:255] if value else None

    @model_validator(mode="after")
    def validate_location(self) -> "Location":
        if self.workingtype == "remote":
            self.worklocation = None
        elif self.workingtype in {"hybrid", "onsite"} and not self.worklocation:
            raise ValueError("worklocation is required for hybrid or onsite.")
        return self


class SoW(BaseModel):
    title: str = Field(max_length=255)
    purpose: str
    definitionOfDone: str
    boundaries: Boundaries
    mustHaveRequirements: List[str] = Field(min_length=1)
    niceToHaveRequirements: List[str] = Field(default_factory=list)
    timeline: str
    budget: Budget
    resources: List[str] = Field(min_length=1)
    location: Location
    language: Language
    type: SowType
    isFinalized: bool = False
    percentage: int = Field(ge=0, le=100)

    @field_validator("title", "purpose", "definitionOfDone", "timeline")
    @classmethod
    def required_non_empty_strings(cls, value: str) -> str:
        value = str(value).strip()
        if not value:
            raise ValueError("Field must not be empty.")
        return value[:255] if value != str(value) else value

    @field_validator("mustHaveRequirements", "niceToHaveRequirements", "resources")
    @classmethod
    def validate_lists(cls, value: List[str]) -> List[str]:
        cleaned = []
        for item in value:
            item = str(item).strip()
            if item:
                cleaned.append(item[:255])
        return cleaned

    @field_validator("timeline")
    @classmethod
    def validate_timeline_format(cls, value: str) -> str:
        value = value.strip()
        if len(value) != 20:
            raise ValueError("timeline must have exact format YYYY-MM-DDYYYY-MM-DD")
        start = value[:10]
        end = value[10:]
        if (
            len(start) != 10
            or len(end) != 10
            or start[4] != "-"
            or start[7] != "-"
            or end[4] != "-"
            or end[7] != "-"
        ):
            raise ValueError("timeline must have exact format YYYY-MM-DDYYYY-MM-DD")
        return value

    @model_validator(mode="after")
    def validate_budget_and_finalization(self) -> "SoW":
        if not self.mustHaveRequirements:
            raise ValueError("mustHaveRequirements must not be empty.")
        if not self.resources:
            raise ValueError("resources must not be empty.")

        if self.type == "materialbased":
            if self.budget.costestimate is None or self.budget.costestimate < 30000:
                raise ValueError("materialbased projects require costestimate >= 30000.")
        elif self.type == "timebased":
            if self.budget.hourlyrate is None or self.budget.hourlyrate < 3000:
                raise ValueError("timebased projects require hourlyrate >= 3000.")
            if (
                self.budget.averageweeklyhours is None
                or self.budget.averageweeklyhours < 4
                or self.budget.averageweeklyhours > 40
            ):
                raise ValueError("timebased projects require averageweeklyhours between 4 and 40.")

        if self.isFinalized:
            required_strings = [
                self.title,
                self.purpose,
                self.definitionOfDone,
                self.timeline,
            ]
            if any(not str(v).strip() for v in required_strings):
                raise ValueError("Cannot finalize with empty required string fields.")
            if not self.boundaries.includedActivities:
                raise ValueError("Cannot finalize with empty includedActivities.")
            if not self.boundaries.outOfScope:
                raise ValueError("Cannot finalize with empty outOfScope.")
            if not self.mustHaveRequirements:
                raise ValueError("Cannot finalize with empty mustHaveRequirements.")
            if not self.resources:
                raise ValueError("Cannot finalize with empty resources.")

        return self


class AssistantResponse(BaseModel):
    displayMessage: str
    SoW: SoW

    @field_validator("displayMessage")
    @classmethod
    def validate_display_message(cls, value: str) -> str:
        value = str(value).strip()
        if not value:
            raise ValueError("displayMessage must not be empty.")
        return value