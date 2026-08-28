# secrets/

Google Cloud 서비스 계정 JSON 을 여기에 둔다.

```
secrets/gcp.json
```

컨테이너에는 `/app/secrets` 로 read-only 마운트되므로
`.env` 에 아래처럼 지정한다.

```
GOOGLE_APPLICATION_CREDENTIALS=/app/secrets/gcp.json
GOOGLE_CLOUD_PROJECT=<프로젝트 ID>
```

이 디렉터리의 JSON 은 `.gitignore` 로 제외된다. 절대 커밋하지 않는다.
