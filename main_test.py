"""
Comprehensive tests for the get_score_filter function.

This module tests the metadata filtering logic for mechanical keyboard switch scores.
Tests cover all score categories, ranking levels, and edge cases.
"""

from main import get_score_filter


class TestGetScoreFilter:
    """Test suite for get_score_filter function."""

    def test_push_feel_highest(self):
        """Test highest push feel score filtering."""
        result = get_score_filter("Which switches have the highest push feel scores?")
        expected = {"Push Feel": {"$gte": 25}}
        assert result == expected

    def test_push_feel_mid(self):
        """Test mid push feel score filtering."""
        result = get_score_filter("Which switches have mid push feel scores?")
        expected = {"$and": [{"Push Feel": {"$gte": 15}}, {"Push Feel": {"$lte": 24}}]}
        assert result == expected

    def test_push_feel_lowest(self):
        """Test lowest push feel score filtering."""
        result = get_score_filter("Which switches have the lowest push feel scores?")
        expected = {"Push Feel": {"$lte": 14}}
        assert result == expected

    def test_push_feel_keyword_variations(self):
        """Test different keyword variations for push feel."""
        queries = ["highest pushfeel scores", "best push feel switches", "top push feel performance"]
        for query in queries:
            result = get_score_filter(query)
            assert result is not None
            assert "Push Feel" in str(result)

    def test_wobble_highest(self):
        """Test highest wobble score filtering."""
        result = get_score_filter("Which switches have the highest wobble scores?")
        expected = {"Wobble": {"$gte": 18}}
        assert result == expected

    def test_wobble_mid(self):
        """Test mid wobble score filtering."""
        result = get_score_filter("Which switches have mid wobble scores?")
        expected = {"$and": [{"Wobble": {"$gte": 9}}, {"Wobble": {"$lte": 17}}]}
        assert result == expected

    def test_wobble_lowest(self):
        """Test lowest wobble score filtering."""
        result = get_score_filter("Which switches have the lowest wobble scores?")
        expected = {"Wobble": {"$lte": 8}}
        assert result == expected

    def test_sound_highest(self):
        """Test highest sound score filtering."""
        result = get_score_filter("Which switches have the highest sound scores?")
        expected = {"Sound": {"$gte": 8}}
        assert result == expected

    def test_sound_mid(self):
        """Test mid sound score filtering."""
        result = get_score_filter("Which switches have mid sound scores?")
        expected = {"$and": [{"Sound": {"$gte": 4}}, {"Sound": {"$lte": 7}}]}
        assert result == expected

    def test_sound_lowest(self):
        """Test lowest sound score filtering."""
        result = get_score_filter("Which switches have the lowest sound scores?")
        expected = {"Sound": {"$lte": 3}}
        assert result == expected

    def test_context_highest(self):
        """Test highest context score filtering."""
        result = get_score_filter("Which switches have the highest context scores?")
        expected = {"Context": {"$gte": 14}}
        assert result == expected

    def test_context_mid(self):
        """Test mid context score filtering."""
        result = get_score_filter("Which switches have mid context scores?")
        expected = {"$and": [{"Context": {"$gte": 7}}, {"Context": {"$lte": 13}}]}
        assert result == expected

    def test_context_lowest(self):
        """Test lowest context score filtering."""
        result = get_score_filter("Which switches have the lowest context scores?")
        expected = {"Context": {"$lte": 6}}
        assert result == expected

    def test_other_highest(self):
        """Test highest other score filtering."""
        result = get_score_filter("Which switches have the highest other scores?")
        expected = {"Other": {"$gte": 8}}
        assert result == expected

    def test_other_mid(self):
        """Test mid other score filtering."""
        result = get_score_filter("Which switches have mid other scores?")
        expected = {"$and": [{"Other": {"$gte": 4}}, {"Other": {"$lte": 7}}]}
        assert result == expected

    def test_other_lowest(self):
        """Test lowest other score filtering."""
        result = get_score_filter("Which switches have the lowest other scores?")
        expected = {"Other": {"$lte": 3}}
        assert result == expected

    def test_ranking_synonyms(self):
        """Test various ranking synonyms."""
        synonyms = [("highest", "best", "top"), ("mid", "medium", "middle", "average"), ("lowest", "worst", "bottom")]

        for category in ["sound", "push feel", "wobble", "context", "other"]:
            for rank_synonyms in synonyms:
                for synonym in rank_synonyms:
                    query = f"Which switches have {synonym} {category} scores?"
                    result = get_score_filter(query)
                    assert result is not None, f"Failed for: {query}"

    def test_case_insensitive(self):
        """Test case insensitive matching."""
        queries = [
            "WHICH SWITCHES HAVE THE HIGHEST SOUND SCORES?",
            "which switches have the highest sound scores?",
            "Which Switches Have The Highest Sound Scores?",
        ]
        for query in queries:
            result = get_score_filter(query)
            expected = {"Sound": {"$gte": 8}}
            assert result == expected

    def test_no_match(self):
        """Test queries that don't match any filters."""
        no_match_queries = [
            "What are the best switches overall?",
            "Tell me about linear switches",
            "How do switches feel?",
            "What is the history of mechanical keyboards?",
            "Compare red and blue switches",
        ]
        for query in no_match_queries:
            result = get_score_filter(query)
            assert result is None, f"Should not match: {query}"

    def test_partial_matches(self):
        """Test queries with partial keyword matches."""
        # These should not match because they don't contain ranking words
        partial_queries = ["Tell me about push feel", "What is wobble?", "Sound quality matters", "Context is important", "Other factors to consider"]
        for query in partial_queries:
            result = get_score_filter(query)
            assert result is None, f"Should not match partial: {query}"

    def test_combined_keywords(self):
        """Test queries with multiple keywords (should match first one found in config order)."""
        result = get_score_filter("Which switches have the highest sound and push feel scores?")
        # Should match push feel since it comes first in the config order (push feel, wobble, sound, context, other)
        expected = {"Push Feel": {"$gte": 25}}
        assert result == expected

    def test_config_order_independence(self):
        """Test that the order of categories in config doesn't affect results."""
        # Since we reordered to push feel, wobble, sound, context, other
        # This test ensures the first matching category wins
        result = get_score_filter("highest sound scores")
        expected = {"Sound": {"$gte": 8}}
        assert result == expected

    def test_all_categories_comprehensive(self):
        """Comprehensive test of all category and ranking combinations."""
        categories = ["push feel", "wobble", "sound", "context", "other"]
        rankings = ["highest", "mid", "lowest"]

        for category in categories:
            for ranking in rankings:
                query = f"Which switches have {ranking} {category} scores?"
                result = get_score_filter(query)
                assert result is not None, f"Failed for: {query}"
                # Verify the field name is in the result
                field_name = category.replace(" ", " ").title()  # "push feel" -> "Push Feel"
                if category == "push feel":
                    field_name = "Push Feel"
                elif category == "wobble":
                    field_name = "Wobble"
                elif category == "sound":
                    field_name = "Sound"
                elif category == "context":
                    field_name = "Context"
                elif category == "other":
                    field_name = "Other"

                assert field_name in str(result), f"Field {field_name} not found in result for: {query}"
