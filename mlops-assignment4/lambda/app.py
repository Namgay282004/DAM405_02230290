import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    """
    AWS Lambda handler function for serverless inference.
    Parses feature payload, checks bounds, and calculates class prediction.
    """
    logger.info("Received event: %s", json.dumps(event))
    try:
        # Extract payload body whether called from API Gateway or direct invocation
        if "body" in event and event["body"]:
            payload = json.loads(event["body"]) if isinstance(event["body"], str) else event["body"]
        else:
            payload = event

        features = payload.get("features", [])
        if not features:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": "Missing 'features' list in payload."})
            }

        # Calculate average feature score
        avg_score = sum(features) / len(features)
        prediction_class = 1 if avg_score > 0.5 else 0

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({
                "status": "success",
                "prediction": prediction_class,
                "confidence_score": round(avg_score, 4)
            })
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)})
        }