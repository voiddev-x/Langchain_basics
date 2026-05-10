from pydantic import BaseModel, EmailStr,Field
from typing import Optional
from pydantic_extra_types.phone_numbers import PhoneNumber

class student(BaseModel):
    name: str
    age: Optional[int] = None
    email: EmailStr#need to install pydantic[email]
    # phoneNumber: int = Field() too redundant instead use
    phoneNumber : PhoneNumber

new_student = {"name": "Vijay",
"email": "vijay@papu.com",
"phoneNumber": '+919999999999'  # a work of my imagination if you can't tell
}

#pydantic smart enough to perform implicit type conversion 

student = student(**new_student)

# print(student.age)
# print(student.phoneNumber)
# print(student.email)

student_dict = dict(student)
# print(student_dict['name'])

student_json = student.model_dump_json()
# print(student_json)