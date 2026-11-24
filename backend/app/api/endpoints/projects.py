"""REST API endpoints for Project management and meeting grouping."""
import re
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, func, and_
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.api.deps import get_current_tenant_membership, TenantContext, require_org_admin
from app.models.project import Project
from app.models.meeting import Meeting
from app.models.memory import AuditLog
from app.schemas.project import ProjectCreate, ProjectUpdate, ProjectResponse

router = APIRouter()


def slugify(text: str) -> str:
    """Generate a clean URL-friendly slug from string."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '-', text)
    return text.strip('-') or "project"


@router.post(
    "",
    response_model=ProjectResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Project",
    description="Create a new project workspace within the current organization."
)
def create_project(
    payload: ProjectCreate,
    tenant: TenantContext = Depends(get_current_tenant_membership),
    db: Session = Depends(get_db),
) -> ProjectResponse:
    org_id = tenant.organization.id
    name = payload.name.strip()
    slug = (payload.slug or slugify(name)).lower().strip()

    # Check if slug already exists in this organization
    existing = db.execute(
        select(Project).where(
            Project.organization_id == org_id,
            Project.slug == slug
        )
    ).scalar_one_or_none()
    if existing:
        # Append unique suffix if collision occurs
        base_slug = slug
        count = 1
        while existing:
            slug = f"{base_slug}-{count}"
            existing = db.execute(
                select(Project).where(
                    Project.organization_id == org_id,
                    Project.slug == slug
                )
            ).scalar_one_or_none()
            count += 1

    project = Project(
        organization_id=org_id,
        name=name,
        slug=slug,
        description=payload.description.strip() if payload.description else None,
        created_by_user_id=tenant.user.id,
    )
    db.add(project)

    audit = AuditLog(
        organization_id=org_id,
        user_id=tenant.user.id,
        action="PROJECT_CREATED",
        entity_type="Project",
        entity_id=project.id,
        details=f"Project '{name}' (slug: {slug}) created."
    )
    db.add(audit)
    db.commit()
    db.refresh(project)

    return ProjectResponse(
        id=project.id,
        organization_id=project.organization_id,
        name=project.name,
        slug=project.slug,
        description=project.description,
        created_by_user_id=project.created_by_user_id,
        created_at=project.created_at,
        updated_at=project.updated_at,
        meeting_count=0
    )


@router.get(
    "",
    response_model=List[ProjectResponse],
    summary="List Projects",
    description="List all projects in the current tenant organization with meeting counts."
)
def list_projects(
    tenant: TenantContext = Depends(get_current_tenant_membership),
    db: Session = Depends(get_db),
) -> List[ProjectResponse]:
    org_id = tenant.organization.id

    # Query projects with meeting count
    stmt = (
        select(
            Project,
            func.count(Meeting.id).label("meeting_count")
        )
        .outerjoin(Meeting, and_(Meeting.project_id == Project.id, Meeting.organization_id == org_id))
        .where(Project.organization_id == org_id)
        .group_by(Project.id)
        .order_by(Project.created_at.desc())
    )
    rows = db.execute(stmt).all()

    results: List[ProjectResponse] = []
    for proj, m_count in rows:
        results.append(
            ProjectResponse(
                id=proj.id,
                organization_id=proj.organization_id,
                name=proj.name,
                slug=proj.slug,
                description=proj.description,
                created_by_user_id=proj.created_by_user_id,
                created_at=proj.created_at,
                updated_at=proj.updated_at,
                meeting_count=m_count
            )
        )
    return results


@router.get(
    "/{project_id}",
    response_model=ProjectResponse,
    summary="Get Project Details",
    description="Get details of a specific project."
)
def get_project(
    project_id: str,
    tenant: TenantContext = Depends(get_current_tenant_membership),
    db: Session = Depends(get_db),
) -> ProjectResponse:
    org_id = tenant.organization.id
    stmt = (
        select(
            Project,
            func.count(Meeting.id).label("meeting_count")
        )
        .outerjoin(Meeting, and_(Meeting.project_id == Project.id, Meeting.organization_id == org_id))
        .where(Project.id == project_id, Project.organization_id == org_id)
        .group_by(Project.id)
    )
    row = db.execute(stmt).first()
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found."
        )

    proj, m_count = row
    return ProjectResponse(
        id=proj.id,
        organization_id=proj.organization_id,
        name=proj.name,
        slug=proj.slug,
        description=proj.description,
        created_by_user_id=proj.created_by_user_id,
        created_at=proj.created_at,
        updated_at=proj.updated_at,
        meeting_count=m_count
    )


@router.patch(
    "/{project_id}",
    response_model=ProjectResponse,
    summary="Update Project",
    description="Update project name or description."
)
def update_project(
    project_id: str,
    payload: ProjectUpdate,
    tenant: TenantContext = Depends(get_current_tenant_membership),
    db: Session = Depends(get_db),
) -> ProjectResponse:
    org_id = tenant.organization.id
    proj = db.execute(
        select(Project).where(
            Project.id == project_id,
            Project.organization_id == org_id
        )
    ).scalar_one_or_none()
    if not proj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found."
        )

    if payload.name is not None and payload.name.strip():
        proj.name = payload.name.strip()
        new_slug = slugify(proj.name)
        # Check collision with other project
        collision = db.execute(
            select(Project).where(
                Project.organization_id == org_id,
                Project.slug == new_slug,
                Project.id != project_id
            )
        ).scalar_one_or_none()
        if not collision:
            proj.slug = new_slug

    if payload.description is not None:
        proj.description = payload.description.strip() or None

    db.commit()
    db.refresh(proj)

    m_count = db.execute(
        select(func.count(Meeting.id)).where(Meeting.project_id == proj.id)
    ).scalar() or 0

    return ProjectResponse(
        id=proj.id,
        organization_id=proj.organization_id,
        name=proj.name,
        slug=proj.slug,
        description=proj.description,
        created_by_user_id=proj.created_by_user_id,
        created_at=proj.created_at,
        updated_at=proj.updated_at,
        meeting_count=m_count
    )


@router.delete(
    "/{project_id}",
    summary="Delete Project",
    description="Delete a project. Existing meetings in this project will have project_id set to null."
)
def delete_project(
    project_id: str,
    tenant: TenantContext = Depends(require_org_admin),
    db: Session = Depends(get_db),
):
    org_id = tenant.organization.id
    proj = db.execute(
        select(Project).where(
            Project.id == project_id,
            Project.organization_id == org_id
        )
    ).scalar_one_or_none()
    if not proj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found."
        )

    # Detach meetings
    db.execute(
        Meeting.__table__.update()
        .where(Meeting.project_id == project_id)
        .values(project_id=None)
    )

    audit = AuditLog(
        organization_id=org_id,
        user_id=tenant.user.id,
        action="PROJECT_DELETED",
        entity_type="Project",
        entity_id=proj.id,
        details=f"Project '{proj.name}' deleted."
    )
    db.add(audit)
    db.delete(proj)
    db.commit()

    return {"message": "Project deleted successfully", "id": project_id}
