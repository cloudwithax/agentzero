# Deliberately broken module for self-heal E2E testing with Claude SDK
def calculate_total(prices):
    # Bug fixed: was using undefined variable
    total = sum(price_dict['amount'] for price_dict in prices)
    return total

def format_report(data):
    # Bug fixed: wrong key name 'totl' -> 'total'
    return f"Report: {data['total']}"

class DataProcessor:
    def process(self, items):
        # Bug fixed: list has no convert_to_upper(); use list comprehension
        return [str(item).upper() for item in items]
