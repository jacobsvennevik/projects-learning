def get_activity_details(self, activity_id: int) -> Dict[str, Any]:
    """Fetch detailed activity data from Garmin Connect."""
    try:
        # Fetch the three detail endpoints
        summary = self.garmin.get_activity(activity_id)
        splits = self.garmin.get_activity_splits(activity_id)
        hr_zones = self.garmin.get_activity_hr_in_timezones(activity_id)
        
        return {
            'summary': summary,
            'splits': splits,
            'hr_zones': hr_zones
        }
    except Exception as e:
        print(f"❌ Failed to fetch details for activity {activity_id}: {e}")
        return None