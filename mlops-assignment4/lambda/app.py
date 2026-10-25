import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    """
    AWS Lambda Handler for Serverless Model Inference.
    Expects a JSON event with a 'features' list.
    """
    logger.info("Received event: %s", json.dumps(event))
    
    try:
        # Parse body if payload is stringified (via API Gateway) or a direct dict
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

        # Model Inference Logic
        avg_score = sum(features) / len(features)
        prediction_class = 1 if avg_score > 0.5 else 0

        response_body = {
            "status": "success",
            "prediction": prediction_class,
            "confidence_score": round(avg_score, 4)
        }

        logger.info("Inference completed: %s", response_body)

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps(response_body)
        }

    except Exception as e:
        logger.error("Error during execution: %s", str(e))
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)})
        }