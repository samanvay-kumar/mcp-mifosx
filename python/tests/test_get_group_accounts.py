# Copyright since 2026 Mifos Initiative
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Unit tests for get_group_accounts MCP tool and domain function."""

import asyncio
from unittest.mock import patch

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

import mcp_server
from core.validation_engine import ValidationError, validate_input
from tools.domains.groups import get_group_accounts as get_group_accounts_domain
from tools.mcp_adapter import fineract_client

MOCK_GROUP_ACCOUNTS = {
    "loanAccounts": [
        {
            "id": 101,
            "accountNo": "000000101",
            "productId": 1,
            "productName": "Group Solidary Loan",
            "status": {"id": 300, "code": "loanStatusType.active", "value": "Active"},
            "originalLoan": 5000.0,
            "loanBalance": 3200.0,
        }
    ],
    "savingsAccounts": [
        {
            "id": 201,
            "accountNo": "000000201",
            "productId": 2,
            "productName": "Group Savings",
            "status": {"id": 300, "code": "savingsAccountStatusType.active", "value": "Active"},
            "accountBalance": 1500.0,
        }
    ],
    "memberLoanAccounts": [],
    "memberSavingsAccounts": [],
}


# ── Successful Execution ──────────────────────────────────────────────────


def test_get_group_accounts_success() -> None:
    """Successful retrieval of group accounts calls Fineract and returns the data."""
    fn = getattr(mcp_server.get_group_accounts, "fn", mcp_server.get_group_accounts)
    with patch.object(fineract_client, "execute_get", return_value=MOCK_GROUP_ACCOUNTS) as mock_get:
        result = fn(groupId=12)

    mock_get.assert_called_once_with("groups/12/accounts")
    assert result == MOCK_GROUP_ACCOUNTS
    assert len(result["loanAccounts"]) == 1
    assert result["loanAccounts"][0]["id"] == 101
    assert len(result["savingsAccounts"]) == 1
    assert result["savingsAccounts"][0]["id"] == 201


def test_get_group_accounts_empty_accounts() -> None:
    """Group without accounts returns empty dictionary or empty lists."""
    fn = getattr(mcp_server.get_group_accounts, "fn", mcp_server.get_group_accounts)
    empty_payload = {"loanAccounts": [], "savingsAccounts": []}
    with patch.object(fineract_client, "execute_get", return_value=empty_payload) as mock_get:
        result = fn(groupId=5)

    mock_get.assert_called_once_with("groups/5/accounts")
    assert result == empty_payload


def test_domain_function_direct_call() -> None:
    """The underlying domain function in groups.py directly queries groups/{id}/accounts."""
    with patch.object(fineract_client, "execute_get", return_value=MOCK_GROUP_ACCOUNTS) as mock_get:
        result = get_group_accounts_domain.func(7)

    mock_get.assert_called_once_with("groups/7/accounts")
    assert result == MOCK_GROUP_ACCOUNTS


# ── Invalid Input Scenarios ───────────────────────────────────────────────


@pytest.mark.parametrize("invalid_id", [0, -1, -99])
def test_get_group_accounts_rejects_non_positive_integers(invalid_id: int) -> None:
    """groupId must be a positive integer (> 0)."""
    fn = getattr(mcp_server.get_group_accounts, "fn", mcp_server.get_group_accounts)
    with pytest.raises(ToolError, match="groupId must be a positive integer"):
        fn(groupId=invalid_id)


@pytest.mark.parametrize("invalid_type", [True, False, "12", 3.14, None])
def test_get_group_accounts_rejects_non_integers(invalid_type) -> None:
    """groupId rejects booleans, strings, floats, and None."""
    fn = getattr(mcp_server.get_group_accounts, "fn", mcp_server.get_group_accounts)
    with pytest.raises(ToolError, match="groupId must be a positive integer"):
        fn(groupId=invalid_type)


def test_validation_engine_direct_check() -> None:
    """Direct invocation of validate_input for get_group_accounts."""
    # Positive integer passes
    validate_input("get_group_accounts", {"groupId": 42})

    # Non-positive or non-integer fails
    with pytest.raises(ValidationError, match="groupId must be a positive integer"):
        validate_input("get_group_accounts", {"groupId": 0})

    with pytest.raises(ValidationError, match="groupId must be a positive integer"):
        validate_input("get_group_accounts", {"groupId": -5})

    with pytest.raises(ValidationError, match="groupId must be a positive integer"):
        validate_input("get_group_accounts", {"groupId": True})


# ── API Failure Scenarios ─────────────────────────────────────────────────


def test_get_group_accounts_api_not_found_404() -> None:
    """When Fineract returns 404, safe_tool turns error sentinel into ToolError."""
    fn = getattr(mcp_server.get_group_accounts, "fn", mcp_server.get_group_accounts)
    not_found_response = {"error": "Group with identifier 99999 does not exist", "status_code": 404}
    with patch.object(fineract_client, "execute_get", return_value=not_found_response):
        with pytest.raises(ToolError, match="does not exist"):
            fn(groupId=99999)


def test_get_group_accounts_api_server_error_500() -> None:
    """When Fineract returns 500 error, safe_tool raises ToolError."""
    fn = getattr(mcp_server.get_group_accounts, "fn", mcp_server.get_group_accounts)
    server_error_response = {"error": "Internal Server Error", "status_code": 500}
    with patch.object(fineract_client, "execute_get", return_value=server_error_response):
        with pytest.raises(ToolError, match="Internal Server Error"):
            fn(groupId=12)


def test_get_group_accounts_backend_timeout() -> None:
    """When adapter reports a timeout error sentinel, safe_tool raises ToolError."""
    fn = getattr(mcp_server.get_group_accounts, "fn", mcp_server.get_group_accounts)
    timeout_response = {"error": "Request timed out after 10.0 seconds"}
    with patch.object(fineract_client, "execute_get", return_value=timeout_response):
        with pytest.raises(ToolError, match="timed out"):
            fn(groupId=12)


def test_get_group_accounts_unexpected_exception() -> None:
    """When an unhandled exception occurs in execution, safe_tool catches it as ToolError."""
    fn = getattr(mcp_server.get_group_accounts, "fn", mcp_server.get_group_accounts)
    with patch.object(fineract_client, "execute_get", side_effect=RuntimeError("connection reset")):
        with pytest.raises(ToolError, match="Internal tool execution error"):
            fn(groupId=12)


def test_mcp_dispatch_rejects_invalid_group_ids() -> None:
    """Tool call through FastMCP Client rejects invalid groupId types and never calls execute_get."""

    async def run() -> None:
        async with Client(mcp_server.mcp) as client:
            for bad_val in ["12", 3.5, True, None]:
                with patch.object(fineract_client, "execute_get") as mock_get:
                    with pytest.raises(ToolError):
                        await client.call_tool("get_group_accounts", {"groupId": bad_val})
                    mock_get.assert_not_called()

    asyncio.run(run())


@pytest.mark.parametrize("group_id", [0, -1, True, False])
def test_domain_function_rejects_invalid_group_id(group_id: object) -> None:
    """Domain function returns an error dict and does not call execute_get for non-positive or non-int IDs."""
    with patch.object(fineract_client, "execute_get") as mock_get:
        result = get_group_accounts_domain.func(group_id)
        mock_get.assert_not_called()
        assert isinstance(result, dict) and "error" in result
        assert f"Invalid group_id: {group_id}. Must be a positive integer." in result["error"]
