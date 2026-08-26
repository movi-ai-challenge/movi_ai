from __future__ import annotations

import json

from .backend_client import (
    build_voice_command_url,
    send_voice_command,
    BackendClientError,
)

from .request_mapper import (
    build_backend_request,
)


def main():

    print()
    print("=" * 70)
    print("BACKEND CLIENT TEST")
    print("=" * 70)

    print(
        "URL:",
        build_voice_command_url(),
    )


    request_data = build_backend_request(

        request_id="req-backend-test",

        transcript=(
            "김민수한테 오만원 보내줘"
        ),

        intent="transfer_money",

        entities={
            "recipient_name": "김민수",
            "recipient_bank": None,
            "recipient_account": None,
            "amount": 50000,
            "source_bank": None,
            "source_account": None,
        },
    )


    print()
    print("[REQUEST]")

    print(
        json.dumps(
            request_data,
            ensure_ascii=False,
            indent=2,
        )
    )


    try:

        response = send_voice_command(
            request_data
        )

    except BackendClientError as error:

        print()
        print("[BACKEND ERROR]")
        print(error)

        return


    print()
    print("[RESPONSE]")

    print(
        json.dumps(
            response,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()