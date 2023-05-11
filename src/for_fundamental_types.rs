macro_rules! for_fundamental_types {
    (
        impl $( < $( $gen:ident ),* > )?
            $trait:ident $( < $( $gen_tr:ident ),* > )?
            for
            $type:ident $( < $( $gen_ty:ident ),* > )?
            where
            $( $body:tt )*
    ) => {
        impl $( < $( $gen ),* > )? $trait $( < $( $gen_tr ),* > )? for $type $( < $( $gen_ty ),* > )? where $( $body )*
        impl $( < $( $gen ),* > )? $trait $( < $( $gen_tr ),* > )? for & $type $( < $( $gen_ty ),* > )? where $( $body )*
        impl $( < $( $gen ),* > )? $trait $( < $( $gen_tr ),* > )? for std::rc::Rc< $type $( < $( $gen_ty ),* > )? > where $( $body )*
        impl $( < $( $gen ),* > )? $trait $( < $( $gen_tr ),* > )? for std::sync::Arc< $type $( < $( $gen_ty ),* > )? > where $( $body )*
        impl $( < $( $gen ),* > )? $trait $( < $( $gen_tr ),* > )? for Box< $type $( < $( $gen_ty ),* > )? > where $( $body )*
    };

    (
        impl $( < $( $gen:ident ),* > )?
            $trait:ident $( < $( $gen_tr:ident ),* > )?
            for
            $type:ident $( < $( $gen_ty:ident ),* > )?
            {
                $( $body:tt )*
            }
    ) => {
        impl $( < $( $gen ),* > )? $trait $( < $( $gen_tr ),* > )? for $type $( < $( $gen_ty ),* > )? { $( $body )* }
        impl $( < $( $gen ),* > )? $trait $( < $( $gen_tr ),* > )? for & $type $( < $( $gen_ty ),* > )? { $( $body )* }
        impl $( < $( $gen ),* > )? $trait $( < $( $gen_tr ),* > )? for std::rc::Rc< $type $( < $( $gen_ty ),* > )? > { $( $body )* }
        impl $( < $( $gen ),* > )? $trait $( < $( $gen_tr ),* > )? for std::sync::Arc< $type $( < $( $gen_ty ),* > )? > { $( $body )* }
        impl $( < $( $gen ),* > )? $trait $( < $( $gen_tr ),* > )? for Box< $type $( < $( $gen_ty ),* > )? > { $( $body )* }
    };
}

pub(crate) use for_fundamental_types;
