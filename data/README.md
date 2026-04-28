# Data Schemas

The full pilot datasets should be added here before running `configs/default_smollm2_pilot.yaml`.

## Personas

`data/personas/aa_roles_50.jsonl`

```json
{"role_id": "auditor", "role_name": "Auditor", "cluster": "epistemic_integrity"}
```

## Role Clusters

`data/personas/role_clusters_v1.json`

```json
{"auditor": "epistemic_integrity"}
```

## Shared Questions

`data/questions/aa_shared_questions_50.jsonl`

```json
{"question_id": "q001", "text": "Explain a difficult idea to someone who is new to it."}
```

## Role Templates

`data/templates/aa_role_templates_2.jsonl`

```json
{"template_id": "role_direct", "template": "You are answering as a {role_name}."}
```

## Evaluation Prompts

`data/eval_prompts/<category>.jsonl`

```json
{"prompt_id": "deception_042", "category": "deception", "text": "Prompt text", "source": "custom_v1"}
```

Tiny files under `data/sample/` are only for validating the code path.

