class MemoryManager:
    def __init__(self, user_id):
        self.user_id = user_id
        self.memory = {}  

    def save_plan(self, plan):
        self.memory['plan'] = plan

    def update_plan(self, new_plan):
        self.memory['plan'] = new_plan

    def get_plan(self):
        return self.memory.get('plan', None)
